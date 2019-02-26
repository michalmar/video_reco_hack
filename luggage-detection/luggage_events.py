"""
LUGGAGE LOADING/WITHDRAWAL START/END event detection

module containing functions and analysis of luggage related events

"""

import numpy as np
import cv2
from scipy import stats
import glob
import imutils


# =============================================================================
# PATHS
# =============================================================================
# path to the cropped images (focused to luggage belt)
lb_img_path = "/Users/Marek/Desktop/DataSentics/MS_hack/luggage_belt_img/"

# path to luggage detection images
ld_img_path = "/Users/Marek/Desktop/DataSentics/MS_hack/luggage_detection/"

# =============================================================================
# CONFIGS
# =============================================================================
# min threshold for image absolute difference (base vs image to check)
min_thresh = 10
# min area for luggage contour detection
min_area = 300
# how many pictures to remember (for base image determination)
img_memory = 50
# =============================================================================
# EXTRACTING BASE IMG (LOADING BELT WITHOUT LUGGAGE)
# =============================================================================
img_filenames = glob.glob(lb_img_path + '*.jpg')

# =============================================================================
# TEST ON ALL IMAGES
# =============================================================================
# precalculate the background image
lb_imgs = cv2.imread(lb_img_path + str(340) + ".jpg", 0)[np.newaxis, :, :]
for i in range(341, 390):
    # read the jpg
    img = cv2.imread(lb_img_path + str(i) + ".jpg", 0)
    # blur the image
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lb_imgs = np.concatenate([lb_imgs, img[np.newaxis, :, :]], axis=0)
for i in range(390, 799):
    # load nexy  image to the remembered ones
    to_check_img = cv2.imread(lb_img_path + str(i) + ".jpg")
    to_check_gray = cv2.cvtColor(to_check_img, cv2.COLOR_BGR2GRAY)
    # blur the image
    to_check_blur = cv2.GaussianBlur(to_check_gray, (5, 5), 0)
    # add the image to the stored ones (for the base calculation)
    lb_imgs = np.concatenate([lb_imgs, to_check_blur[np.newaxis, :, :]], axis=0)
    # remove the first (oldest) image
    lb_imgs = lb_imgs[1:, :, :]
    # find base image
    base_img = np.squeeze(stats.mode(lb_imgs, axis=0)[0], axis=0)
    # compute the absolute difference between base and image to check
    frameDelta = cv2.absdiff(base_img, to_check_blur)
    thresh = cv2.threshold(frameDelta, min_thresh, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) >= min_area:
            # compute the bounding box for the contour, draw it on the frame,
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(to_check_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # save the image with bounding box
    cv2.imwrite(ld_img_path + str(i) + ".jpg", to_check_img)
