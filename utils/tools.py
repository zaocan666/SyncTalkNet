import os, subprocess, glob, pandas, tqdm, cv2
import torch
import random
import numpy as np
from scipy.io import wavfile
from multiprocessing import Process
import glog as log
import sys

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    args.trialPath     = os.path.join(args.dataPath, 'json_process')
    args.audioPath     = os.path.join(args.dataPath, 'clips_audios')
    args.visualPath    = os.path.join(args.dataPath, 'clips_videos')
    
    if os.path.exists(args.savePath):
        assert args.ssh==False
        user_del=input(args.savePath+' exists, do you want to delete? (y/n)')
        if user_del=='y' or user_del=='Y' :
            os.system('rm -r '+args.savePath)

    os.makedirs(args.modelSavePath, exist_ok = True)
    # os.makedirs(args.dataPathAVA, exist_ok = True)
    return args
 

def download_pretrain_model_AVA():
    if os.path.isfile('pretrain_AVA.model') == False:
        Link = "1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm"
        cmd = "gdown --id %s -O %s"%(Link, 'pretrain_AVA.model')
        subprocess.call(cmd, shell=True, stdout=None)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def print_args(args):
    log.info('-------- args -----------')
    for k,v in vars(args).items():
        log.info('%s: '%k+str(v))
    log.info('-------------------------')

def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def save_video(pics, audio, out_pth, fps, audio_sr=16000):
    from moviepy.editor import VideoFileClip, AudioFileClip
    # cv2.imwrite('test.png', pics[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    dsize = np.asarray(pics[0]).shape[:2]
    videoWriter  = cv2.VideoWriter(out_pth, fourcc, fps, dsize)
    for i in range(len(pics)):
        pici = cv2.cvtColor(pics[i], cv2.COLOR_GRAY2BGR)
        videoWriter.write(pici)
    videoWriter.release()
    
    video = VideoFileClip(out_pth)
    
    if type(audio)!=type(None):
        audio_tmp_name='tmp.wav'
        wavfile.write(audio_tmp_name, audio_sr, audio)
        videos = video.set_audio(AudioFileClip(audio_tmp_name))  # 音频文件
        # cmd = 'ffmpeg -i %s -i %s -c copy -map 0:v:0 -map 1:a:0 output.avi'%(out_pth, audio_tmp_name)
        # os.system(cmd)
        os.system('rm '+audio_tmp_name)

    videos.write_videofile(out_pth.split('.')[0]+'_out.'+out_pth.split('.')[1], audio_codec='aac')  # 保存合成视频，注意加上参数audio_codec='aac'，否则音频无声音
    os.system('rm '+out_pth)

def preprocess_AVA(args):
    # This preprocesstion is modified based on this [repository](https://github.com/fuankarion/active-speakers-context).
    # The required space is 302 G. 
    # If you do not have enough space, you can delate `orig_videos`(167G) when you get `clips_videos(85G)`.
    #                             also you can delate `orig_audios`(44G) when you get `clips_audios`(6.4G).
    # So the final space is less than 100G.
    # The AVA dataset will be saved in 'AVApath' folder like the following format:
    # ```
    # ├── clips_audios  (The audio clips cut from the original movies)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── clips_videos (The face clips cut from the original movies, be save in the image format, frame-by-frame)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── csv
    # │   ├── test_file_list.txt (name of the test videos)
    # │   ├── test_loader.csv (The csv file we generated to load data for testing)
    # │   ├── test_orig.csv (The combination of the given test csv files)
    # │   ├── train_loader.csv (The csv file we generated to load data for training), 各列：track name, frame num, fps, label, random num?
    # │   ├── train_orig.csv (The combination of the given training csv files)
    # │   ├── trainval_file_list.txt (name of the train/val videos)
    # │   ├── val_loader.csv (The csv file we generated to load data for validation)
    # │   └── val_orig.csv (The combination of the given validation csv files)
    # ├── orig_audios (The original audios from the movies)
    # │   ├── test
    # │   └── trainval
    # └── orig_videos (The original movies)
    #     ├── test
    #     └── trainval
    # ```

    # download_csv(args) # Take 1 minute 
    # download_videos(args) # Take 6 hours
    # extract_audio(args) # Take 1 hour
    # extract_audio_clips(args) # Take 3 minutes
    extract_video_clips(args) # Take about 2 days

def download_csv(args):
    # Take 1 minute to download the required csv files
    Link = "1C1cGxPHaJAl1NQ2i7IhRgWmdvsPhBCUy"
    cmd = "gdown --id %s -O %s"%(Link, args.dataPathAVA + '/csv.tar.gz')
    subprocess.call(cmd, shell=True, stdout=None)
    cmd = "tar -xzvf %s -C %s"%(args.dataPathAVA + '/csv.tar.gz', args.dataPathAVA)
    subprocess.call(cmd, shell=True, stdout=None)
    os.remove(args.dataPathAVA + '/csv.tar.gz')

def download_videos(args):
    # Take 6 hours to download the original movies, follow this repository: https://github.com/cvdfoundation/ava-dataset
    for dataType in ['trainval', 'test']:
        fileList = open('%s/%s_file_list.txt'%(args.trialPathAVA, dataType)).read().splitlines()   
        outFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        os.makedirs(outFolder)

        for fileName in fileList:
            # cmd = "wget -P %s https://s3.amazonaws.com/ava-dataset/%s/%s"%(outFolder, dataType, fileName)
            cmd = "mwget -d %s https://s3.amazonaws.com/ava-dataset/%s/%s"%(outFolder, dataType, fileName)
            subprocess.call(cmd, shell=True, stdout=None)
        

def extract_audio(args):
    # Take 1 hour to extract the audio from movies
    for dataType in ['trainval', 'test']:
        inpFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        outFolder = '%s/%s'%(args.audioOrigPathAVA, dataType)
        os.makedirs(outFolder, exist_ok = True)
        videos = glob.glob("%s/*"%(inpFolder))
        for videoPath in tqdm.tqdm(videos):
            audioPath = '%s/%s'%(outFolder, videoPath.split('/')[-1].split('.')[0] + '.wav')
            cmd = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic" % (videoPath, audioPath))
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio_clips(args):
    # Take 3 minutes to extract the audio clips
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in ['train', 'val', 'test']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv'%(dataType)), engine='python')
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        audioFeatures = {}
        outDir = os.path.join(args.audioPathAVA, dataType)
        audioDir = os.path.join(args.audioOrigPathAVA, dic[dataType])
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            start = insData.iloc[0]['frame_timestamp']
            end = insData.iloc[-1]['frame_timestamp']
            entityID = insData.iloc[0]['entity_id']
            insPath = os.path.join(outDir, videoKey, entityID+'.wav')
            if videoKey not in audioFeatures.keys():                
                audioFile = os.path.join(audioDir, videoKey+'.wav')
                sr, audio = wavfile.read(audioFile)
                audioFeatures[videoKey] = audio
            audioStart = int(float(start)*sr)
            audioEnd = int(float(end)*sr)
            audioData = audioFeatures[videoKey][audioStart:audioEnd]
            wavfile.write(insPath, sr, audioData)

def extract_video_clips_process(df_process, videoDir, outDir):
    entityList = np.sort(df_process['entity_id'].unique()).tolist()
    df_process = df_process.groupby('entity_id')
    for l in df_process['video_id'].unique().tolist():
        d = os.path.join(outDir, l[0])
        if not os.path.isdir(d):
            os.makedirs(d)
    
    current_videoid = ''
    current_V = None

    # for entity in tqdm.tqdm(entityList, total = len(entityList)):
    for i, entity in enumerate(entityList):
        print('process %d, %d/%d'%(os.getpid(), i, len(entityList)), flush=True)
        insData = df_process.get_group(entity)
        videoKey = insData.iloc[0]['video_id']
        entityID = insData.iloc[0]['entity_id']
    
        if current_videoid != videoKey:
            videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(videoKey)))[0]
            current_V = cv2.VideoCapture(videoFile)
            current_videoid = videoKey

        insDir = os.path.join(os.path.join(outDir, videoKey, entityID))
        if not os.path.isdir(insDir):
            os.makedirs(insDir)
        j = 0
        for _, row in insData.iterrows():
            imageFilename = os.path.join(insDir, str("%.2f"%row['frame_timestamp'])+'.jpg')
            current_V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
            _, frame = current_V.read()
            h = np.size(frame, 0)
            w = np.size(frame, 1)
            x1 = int(row['entity_box_x1'] * w)
            y1 = int(row['entity_box_y1'] * h)
            x2 = int(row['entity_box_x2'] * w)
            y2 = int(row['entity_box_y2'] * h)
            face = frame[y1:y2, x1:x2, :]
            j = j+1
            cv2.imwrite(imageFilename, face)

def extract_video_clips(args):
    # Take about 2 days to crop the face clips.
    # You can optimize this code to save time, while this process is one-time.
    # If you do not need the data for the test set, you can only deal with the train and val part. That will take 1 day.
    # This procession may have many warning info, you can just ignore it.
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    # for dataType in ['train', 'val', 'test']:
    for dataType in ['train', 'val']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv'%(dataType)))
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        outDir = os.path.join(args.visualPathAVA, dataType)
        videoDir = os.path.join(args.visualOrigPathAVA, dic[dataType])

        # group the video_id and df rows 
        process_num = 30
        videoidList = df['video_id'].unique().tolist()
        video_eachprocess = (len(videoidList)+process_num-1)//process_num
        videoList_process = [videoidList[i*video_eachprocess:(i+1)*video_eachprocess] for i in range(process_num)] # video list for each process
        dfList_process = [df[df['video_id'].isin(videoList_process[i])] for i in range(process_num)]
        print('total_row : %d'%sum([dfi.shape[0] for dfi in dfList_process]), flush=True)
        processList = [Process(target=extract_video_clips_process, args=(dfi, videoDir, outDir)) for dfi in dfList_process]
        for proc in processList:
            proc.start()
        for proc in processList:
            proc.join()

        
