# video_reco_hack 

## components
- added split video into frames
- added yolo v3 pre-trained coco video inference

## install

1. clone repo

    `git clone https://github.com/michalmar/video_reco_hack.git`

1. [recommended] create separate conda environment

    `conda create --name py36vidhack python=3.6`
    
    `source activate py36vidhack`
1. install requirements (opencv, numpy, imutils)

    `pip install -r requirements.txt`

1. run the sample
    
    `cd video_reco_hack/yolo-object-detection`

    `python yolo_video.py --input videos/videoplayback-320p-short.mp4 --output output/videoplayback-320p-short.avi --yolo yolo-coco`