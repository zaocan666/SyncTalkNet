import time, os, torch, argparse, warnings
import glog as log
import glob

from dataLoader import ASW_loader
from utils.tools import init_args, set_log_file, setup_seed, preprocess_AVA, print_args
from talkNet import talkNet

def get_model(args):
    return talkNet(lr = args.lr, lrDecay = args.lrDecay, positional_emb_flag=args.positional_emb_flag,
                    track_exchange = args.track_exchange)

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=24,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=3,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--track_max_len',    type=int,   default=250,  help='track_max_len')
    parser.add_argument('--nDataLoaderThread', type=int, default=10,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPath',  type=str, default="", help='Save path dataset')
    parser.add_argument('--savePath',     type=str, default="train_output")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # Sync revision
    parser.add_argument('--positional_emb_flag', type=bool, default=True,  help='Whether to use positional embedding')
    parser.add_argument('--track_exchange', type=bool, default=True,  help='Whether to use track exchanging negative samples')
    # For download dataset only, for evaluation only
    parser.add_argument('--evaluation', type=bool, default=False, help='evaluate or not')
    parser.add_argument('--evaluation_pth', type=str, default='', help='evaluated model pth')
    parser.add_argument('--syncDestroy', type=str, default="none", help='way of synchronization corruption. [none, audioShift, audioFlip, audioRepeat, audioSwap]')
    parser.add_argument('--audio_change_ratio', type=float, default=0, help='proportion of audio speaking segment being corrupted')
    parser.add_argument('--random_seed', type=int, default=1234, help='random_seed')
    parser.add_argument('--ssh', action='store_true', help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')
    args = parser.parse_args()
    args = init_args(args)
    print_args(args)
    setup_seed(args.random_seed)

    set_log_file(os.path.join(args.savePath, 'run.log'), file_only=args.ssh)

    if not args.evaluation:
        assert args.syncDestroy=='none'

    train_loader = ASW_loader(trialPath = args.trialPath, \
                          audioPath = os.path.join(args.audioPath , 'train'), \
                          visualPath = os.path.join(args.visualPath, 'train'), \
                          batchSize = args.batchSize, track_max_len=args.track_max_len, \
                          train_flag=True, syncDestroy='none', \
                         audio_change_ratio=0)
    trainLoader = torch.utils.data.DataLoader(train_loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    val_loader = ASW_loader(trialPath = args.trialPath, \
                          audioPath = os.path.join(args.audioPath , 'val'), \
                          visualPath = os.path.join(args.visualPath, 'val'), \
                          batchSize = 1, track_max_len=args.track_max_len, \
                          train_flag=False, syncDestroy=args.syncDestroy, \
                         audio_change_ratio=args.audio_change_ratio)
    valLoader = torch.utils.data.DataLoader(val_loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    if args.evaluation == True:
        s = get_model(args)
        s.loadParameters(args.evaluation_pth)
        log.info("Model %s loaded from previous state!"%(args.evaluation_pth))
        epoch_auc, epoch_ap, epoch_acc = s.evaluate_network(epoch = 0, loader = valLoader)
        log.info("Eval, Acc %2.2f%%, AUC: %2.2f%%, mAP %2.2f%%"%(epoch_acc, epoch_auc, epoch_ap))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()
    if len(modelfiles) >= 1:
        log.info("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = get_model(args)
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = get_model(args)

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:
            epoch_auc, epoch_ap, epoch_acc = s.evaluate_network(epoch = epoch, loader = valLoader)
            mAPs.append(epoch_ap)

            if mAPs[-1]==max(mAPs):
                modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
                for f in modelfiles:
                    os.system('rm '+f)
                s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)

            log.info("Eval %d epoch, Acc %2.2f%%, AUC: %2.2f%%, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, epoch_acc, epoch_auc, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, Acc %2.2f%%, AUC: %2.2f%%, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, epoch_acc, epoch_auc, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
