'''
Using OpenCV takes a mp4 video and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

import datetime  
import time

# vid_file = "videoplayback-720p-short"
vid_file = "videoplayback-720p"
# Playing video from file:
cap = cv2.VideoCapture('../data/'+vid_file+'.mp4')

try:
    if not os.path.exists('../data'):
        os.makedirs('../data')
except OSError:
    print ('Error: Creating directory of data')


# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
   
if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
myTime = datetime.datetime(2019, 2, 1)


currentFrame = 0
while(currentFrame < 10000):
    # Capture frame-by-frame
    ret, frame = cap.read()
#     print("{} {}".format(currentFrame, currentFrame % 25))
    if (currentFrame % 25 == 0):
        myTime = myTime + datetime.timedelta(0,1) # days, seconds, then other fields.
    # Saves image of the current frame in jpg file
    name = '../data/'+vid_file+'_' + str(currentFrame).zfill(5) + '_at_'+ myTime.strftime("%H-%M-%S") +'.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()