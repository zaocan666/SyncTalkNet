## Rethinking Audio-visual Synchronization for Active Speaker Detection

## Usage
### dataset
The preparation of AVA-Activespeaker dataset follows https://github.com/TaoRuijie/TalkNet_ASD

### Training
python trainTalkNet.py --dataPath /path/to/data/

### Eval
python trainTalkNet.py --dataPath /path/to/data/ --evaluation True --evaluation_pth /path/to/model

### Unsynchronization eval
python trainTalkNet.py --dataPath /path/to/data/ --evaluation True --evaluation_pth /path/to/model --syncDestroy audioSwap --audio_change_ratio 0.5

## Reference
https://github.com/TaoRuijie/TalkNet_ASD

https://github.com/clovaai/lookwhostalking
