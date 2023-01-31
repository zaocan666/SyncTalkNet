from argparse import ArgumentError
import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
import glog as log
import json
import numpy as np

from utils.tools import save_video

fps = 25.0

def get_track_audio(track, audioPath):
    videoName = track['video_name']
    trackName = track['track_name'].split('|')[0]

    audio_offset = track['track_start_time']
    min_ts = float(track['data'][0]['time'])
    max_ts = float(track['data'][-1]['time'])
    audio_pth = os.path.join(audioPath, videoName, trackName + '.wav')
    sample_rate, audio_data = wavfile.read(audio_pth)
    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]
    return audio_clip

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for track in batchList:
        audio = get_track_audio(track, dataPath)
        audioSet[track['video_name']+'|'+track['track_name']] = audio
    return audioSet

def overlap(dataName, audio, audioSet):
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

def load_audio(track, dataPath, numFrames, audioAug, audioSet = None):
    dataName = track['video_name']+'|'+track['track_name']

    audio = audioSet[dataName]    
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4) # audio frame is 10*25/fps ms long, visual frame is 1000/fps ms long
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio

def load_visual(track, dataPath, numFrames, visualAug):
    faceFolderPath = os.path.join(dataPath, track['video_name'], track['track_name'].split('|')[0])
    faceFiles = [os.path.join(faceFolderPath, '%06d.png'%frame_info['frame_index']) for frame_info in track['data']]
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (int(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces

def load_label(track, numFrames):
    labels = [frame_info['label'] for frame_info in track['data']]
    res = numpy.array(labels[:numFrames])
    return res

class ASW_loader(object):
    def __init__(self, trialPath, audioPath, visualPath, batchSize, track_max_len, train_flag, syncDestroy, audio_change_ratio):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []
        self.train_flag = train_flag
        self.syncDestroy = syncDestroy
        self.audio_change_ratio = audio_change_ratio

        if self.train_flag:
            assert self.syncDestroy=='none'

        trackList = self._parse_json(trialPath, visualPath)
        trackList_split = self._split_track(trackList, track_max_len)

        if not train_flag:
            assert batchSize==1
            self.miniBatch = [[track] for track in trackList_split]
            return

        random.shuffle(trackList_split)
        sortedTrackList = sorted(trackList_split, key=lambda track: len(track['data']), reverse=True)  

        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        start = 0
        while True:
          length = len(sortedTrackList[start]['data'])
          end = min(len(sortedTrackList), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedTrackList[start:end])
          if end == len(sortedTrackList):
              break
          start = end     

    def _parse_json(self, json_dir, video_root):
        video_names = [os.path.basename(pth) for pth in glob.glob(video_root+'/*')]
        json_list = [os.path.join(json_dir, vn+'.json') for vn in video_names]
        log.info('video num: %d'%(len(json_list))) 
        
        track_list = []

        for json_f in json_list:
            video_name = os.path.basename(json_f).split('.')[0]

            with open(json_f) as f:
                track_dict = json.load(f)
                track_names = sorted(track_dict.keys())
                
                for track in track_names:
                    track_list.append({'video_name':video_name, 'track_name':track, 'data':track_dict[track]})
        
        log.info('track  num: %d'%(len(track_list))) # train 1338129

        return track_list

    def _split_track(self, trackList, track_max_len):
        track_output = []
        for track_info in trackList:
            track_len = len(track_info['data'])
            if track_len<=track_max_len and track_len>0:
                track_info_out = track_info
                track_info_out['track_start_time']=track_info['data'][0]['time']
                track_output.append(track_info)
                continue
            
            assert '|' not in track_info['track_name']
            split_track_num = (track_len+track_max_len-1)//track_max_len
            splited_tracks = []
            sum_frames = 0
            i=0
            while i < split_track_num:
                if i==split_track_num-2 and (track_len-(split_track_num-1)*track_max_len)<=round(track_max_len*0.5):
                    end_index = track_len
                    split_track_num -= 1
                else:
                    end_index = (i+1)*track_max_len

                track_split_i = {'track_start_time':track_info['data'][0]['time'],
                                'video_name':track_info['video_name'], 
                                'track_name':track_info['track_name']+'|%d'%i,
                                'data':track_info['data'][i*track_max_len:end_index]}
                splited_tracks.append(track_split_i)
                sum_frames += len(track_split_i['data'])
                i += 1
            assert sum_frames==track_len
            track_output += splited_tracks
        return track_output

    def __getitem__(self, index):
        batchList    = self.miniBatch[index] # a batch where each item is a face-track
        numFrames   = len(batchList[-1]['data']) # numFrame of the shortest segment
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for track in batchList:
            visualFeatures.append(load_visual(track, self.visualPath, numFrames, visualAug = self.train_flag))
            labels.append(load_label(track, numFrames))

        if self.syncDestroy=='audioSwap':
            while True:
                index_j = np.random.randint(0, self.__len__())
                track_j = self.miniBatch[index_j][0]
                label_j = load_label(track_j, len(track_j['data']))
                if (track_j['video_name']==batchList[0]['video_name']) or (not (1 in label_j)):
                    continue
                audio_j = generate_audio_set(self.audioPath, [track_j])[track_j['video_name']+'|'+track_j['track_name']]
                break

        if self.syncDestroy.startswith('audio'):
            track = batchList[0]
            audio_destroy, label_update = audio_change(audioSet[track['video_name']+'|'+track['track_name']], labels[0], destroy_type=self.syncDestroy,
                                         change_ratio=self.audio_change_ratio, fps=fps, pics=visualFeatures[0],
                                         audio_j=audio_j, label_j=label_j, fps_j=fps)
            audioSet = {track['video_name']+'|'+track['track_name']:audio_destroy}
            labels = [label_update]
        elif self.syncDestroy=='visual':
            raise NotImplementedError()

        for track in batchList:
            audioFeatures.append(load_audio(track, self.audioPath, numFrames, audioAug = self.train_flag, audioSet = audioSet))  

        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)

def find_speaking_segment(label, fps, audio_sr):
    isone = np.concatenate(([0], label.view(np.int), [0]))
    absdiff = np.abs(np.diff(isone))
    speak_ranges = np.where(absdiff == 1)[0].reshape(-1, 2) # [[start1, end1], [start2, end2],...] start:end is speaking segment
    speak_ranges_time = speak_ranges/float(fps)
    speak_ranges_audio = np.round(speak_ranges_time*audio_sr).astype(int) # audio index
    return speak_ranges, speak_ranges_time, speak_ranges_audio

def audio_change(audio, label, destroy_type, change_ratio, fps, pics, audio_j=None, label_j=None, fps_j=None, audio_sr=16000):
    # find speaking segments through label
    speak_ranges, speak_ranges_time, speak_ranges_audio = find_speaking_segment(label, fps, audio_sr)

    audio_destroy = audio.copy()
    destroy_time = []
    for i in range(speak_ranges_audio.shape[0]):
        if np.random.rand()>change_ratio:
            continue

        start = int(speak_ranges_audio[i,0])
        if start>=audio.shape[0]:
            continue
        end = min(speak_ranges_audio[i,1], audio.shape[0])

        if destroy_type=='audioShift':
            shift_len = (end-start)//2
            audio_destroy[start:end] = np.roll(audio[start:end], shift_len, axis=0)
        elif destroy_type=='audioFlip':
            audio_destroy[start:end]=audio[start:end][::-1]
        elif destroy_type=='audioRepeat':
            repeat_len = min((end-start)//3, int(0.8*audio_sr))
            segment_start = np.random.choice(range(start, end-repeat_len))
            segment_repeated = np.tile(audio[segment_start:segment_start+repeat_len], (end-start)//repeat_len+1)
            audio_destroy[start:end]=segment_repeated[:end-start]
        elif destroy_type=='audioSilence':
            audio_destroy[start:end]=0
        elif destroy_type=='audioSwap':
            _, _, speak_ranges_audio_j = find_speaking_segment(label_j, fps_j, audio_sr)
            audio_j_start = speak_ranges_audio_j[0,0]
            audio_j_end = min(speak_ranges_audio_j[0,1], audio_j.shape[0])

            audio_j_repeated = np.tile(audio_j[audio_j_start:audio_j_end], (end-start)//(audio_j_end-audio_j_start)+1)
            audio_destroy[start:end]=audio_j_repeated[:end-start]
        else:
            raise ArgumentError('destroy_type wrong')

        label[speak_ranges[i,0]: speak_ranges[i,1]]=0
        destroy_time.append(speak_ranges_time[i])
    
    return audio_destroy, label