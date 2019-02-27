# video_reco_hack 

## components
- added split video into frames
- added yolo v3 pre-trained coco video inference (only)
- added Faster-RCNN algo using VGG16 net

## install

1. clone repo

    `git clone https://github.com/michalmar/video_reco_hack.git`

1. [recommended] create separate conda environment

    `conda create --name py36vidhack python=3.6`
    
    `source activate py36vidhack`
1. install requirements (opencv, numpy, imutils)

    `pip install -r requirements.txt`

##  Faster-RCNN algo using VGG16 net
all relevant code and aretefacts are in folder: `faster-rcnn_vgg16` -> every steps are with relative path to that{!}

### training
download starting weights into ./model/: `wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5`

[option 1] run `python ./train.py`

[option 2] interactive Jupyter notebook: [frcnn_train_vgg.ipynb](./frcnn_train_vgg.ipynb)



### testing / scoring
assuming you have built your model
interactive Jupyter notebook: [frcnn_test_vgg.ipynb](./frcnn_test_vgg.ipynb)


## yolo v3 pre-trained coco video inference (only)

1. run the sample
    
    `cd video_reco_hack/yolo-object-detection`

    `wget https://pjreddie.com/media/files/yolov3.weights`

    `python yolo_video.py --input videos/videoplayback-320p-short.mp4 --output output/videoplayback-320p-short.avi --yolo yolo-coco`



## TODO:
- [ ] add luggage detection on belt
- [x] create video from detections for all images 
- [ ] train with additional data


