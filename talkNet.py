import torch
import torch.nn as nn
import torch.nn.functional as F
import glog as log
import sys, time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import glog as log
import numpy as np

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, positional_emb_flag=False, track_exchange=False):
        super(talkNet, self).__init__()
        self.model = talkNetModel(positional_emb_flag).cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        self.track_exchange = track_exchange
        log.info(" Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()

        log.info('')
        log.info('Epoch {}'.format(epoch))
        log.info('-' * 10)

        self.scheduler.step(epoch - 1)
        log.info('lr: %.3e'%self.optim.state_dict()['param_groups'][0]['lr'])

        index, top1, loss, loss_track_change = 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        start_time = time.time()
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            audioFeature = audioFeature[0].cuda()
            visualFeature = visualFeature[0].cuda()
            labels = labels[0].cuda() # [batchsize, numFrames]
            data_time = time.time()-start_time
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature)
            visualEmbed = self.model.forward_visual_frontend(visualFeature)

            track_change_start_ind = audioEmbed.shape[0]
            if self.track_exchange:
                labels_positive_ind = torch.where(labels.sum(1)>0)[0]
                if len(labels_positive_ind)>1:
                    audioEmbed_postive = audioEmbed[labels_positive_ind]
                    visualEmbed_positive = visualEmbed[labels_positive_ind]
                    
                    # shuffle the visual embd
                    audio_ind = np.arange(audioEmbed_postive.shape[0])
                    visual_ind = np.arange(audioEmbed_postive.shape[0])
                    while (visual_ind==audio_ind).any():
                        np.random.shuffle(visual_ind)

                    visualEmbed_positive = visualEmbed_positive[visual_ind]

                    audioEmbed = torch.cat([audioEmbed, audioEmbed_postive], dim=0)
                    visualEmbed = torch.cat([visualEmbed, visualEmbed_positive], dim=0)
                    labels = torch.cat([labels, torch.zeros((len(labels_positive_ind), labels.shape[1]), dtype=torch.long).cuda()], dim=0)

            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
            nlossAV, nloss_track_change, _, _, prec = self.lossAV.forward(outsAV, labels.reshape(-1), track_change_start_ind*audioEmbed.shape[1])

            outsA = self.model.forward_audio_backend(audioEmbed[:track_change_start_ind])
            outsV = self.model.forward_visual_backend(visualEmbed[:track_change_start_ind])
            nlossA = self.lossA.forward(outsA, labels[:track_change_start_ind].reshape(-1))
            nlossV = self.lossV.forward(outsV, labels[:track_change_start_ind].reshape(-1))

            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV

            loss += nloss.detach().cpu().numpy()
            loss_track_change += nloss_track_change
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += labels.shape[0]*labels.shape[1]

            batch_time = time.time()-start_time
            start_time = time.time()
            if num%(len(loader)//30)==0:
                log.info('Train iter {:d}/{:d} batchTime: {:.1f} dataTime: {:.1f} | Loss: {:.4f} Loss_track_change: {:.4f} | ACC: {:.2f}%%'.format(
                    num, len(loader), batch_time, data_time, loss/num, loss_track_change/num, 100 * (top1/index)))
            
        return loss/num, lr

    def evaluate_network(self, epoch, loader):
        self.eval()
        predScores = []
        label_lst = []
        start_time = time.time()
        index, top1 = 0, 0
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            with torch.no_grad():
                audioFeature = audioFeature[0].cuda()
                visualFeature = visualFeature[0].cuda()
                labels = labels[0].cuda()
                data_time = time.time()-start_time
                audioEmbed  = self.model.forward_audio_frontend(audioFeature)
                visualEmbed = self.model.forward_visual_frontend(visualFeature)
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels.reshape((-1))            
                _, _, predScore, _, prec = self.lossAV.forward(outsAV, labels, audioEmbed.shape[0]) 
                top1 += prec   
                index += len(labels)

                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                label_lst.extend(labels.cpu().numpy().tolist())

                batch_time = time.time()-start_time
                start_time = time.time()
                if num%(len(loader)//30)==0:
                    log.info('Eval iter {:d}/{:d} batchTime: {:.1f} dataTime: {:.1f} | ACC: {:.2f}%%'.format(
                        num, len(loader), batch_time, data_time, 100 * (top1/index)))

        epoch_auc = roc_auc_score(label_lst, predScores)
        epoch_ap = average_precision_score(label_lst, predScores)

        return epoch_auc*100, epoch_ap*100, 100 * (top1/index)

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    log.info("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
