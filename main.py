# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.ImageProcessor import *

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"
DEBUG = True
NAME_PICTURE = 'lena2.jpg'

s = 3
octave = 3


# Algorithm
img = ImageManager.loadMatrix(ORIGINAL_IMAGES_PATH + NAME_PICTURE)
ImageProcessor.findKeypoints(img, s, octave, verbose=DEBUG)


