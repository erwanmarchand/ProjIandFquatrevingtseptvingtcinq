# -*- coding: utf-8 -*-
from lib.ExtremaDetector import ExtremaDetector
from lib.ImageManager import *

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"

NAME_PICTURE = 'lena2.jpg'
s = 3
octave = 1

img = ImageManager.loadMatrix(ORIGINAL_IMAGES_PATH + NAME_PICTURE)
ExtremaDetector.differenceDeGaussienne(img, s, octave, verbose=True)


