"""
LUGGAGE BELT IMAGE CROPPING

script for cropping images -> resulting images contain only luggage belt

"""

import cv2
import glob

# =============================================================================
# PATHS
# =============================================================================
# path to the images from the main 720p video
img_path = "/Users/Marek/Desktop/DataSentics/MS_hack/data_720p/"
# path to the cropped images (focused to luggage belt)
lb_img_path = "/Users/Marek/Desktop/DataSentics/MS_hack/luggage_belt_img/"

# =============================================================================
# LOAD IMG, CROP LUGGAGE BELT, SAVE IMG
# =============================================================================
img_filenames = glob.glob(img_path + 'videoplayback-720p-short*.jpg')
for i in range(len(img_filenames)):
    # read the jpg
    img = cv2.imread(img_filenames[i])
    img_crop = img[450:550, 290:550]
    cv2.imwrite(lb_img_path + str(i) + ".jpg", img_crop)