# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.ImageProcessor import *
import cv2

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"
DEBUG = True
SHOW_IMAGE = False
NAME_PICTURE = 'lena.jpg'

s = 3
octave = 3


# Algorithm
img = ImageManager.loadMatrix(ORIGINAL_IMAGES_PATH + NAME_PICTURE)
keypoints = ImageProcessor.findKeypoints(img, s, octave, verbose=DEBUG, show_images=SHOW_IMAGE)
beacons = ImageProcessor.showKeyPoints(img, keypoints)

ImageManager.showImage(beacons, cmap='gray')

