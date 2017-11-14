# -*- coding: utf-8 -*-
from lib.ExtremaDetector import ExtremaDetector
from lib.Image import *

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"

NAME_PICTURE = 'einstein.jpg'
sigma = 1.6
octave = 1

maxI = 3
maxk = 5

img = Image.loadMatrix(ORIGINAL_IMAGES_PATH + NAME_PICTURE)
ExtremaDetector.differenceDeGaussienne(img, 3, 1, verbose=True)


