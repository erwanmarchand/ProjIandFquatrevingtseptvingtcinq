# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
import cv2

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"
DEBUG = True
SHOW_IMAGE = False
NAME_PICTURE = 'lena.jpg'

s = 3
octave = 3

#  On charge l'image et on la redimensionne
imgWithColor = ImageManager.loadMatrix(ORIGINAL_IMAGES_PATH + NAME_PICTURE)
imgWithColor = ImageManager.getOctave(imgWithColor, -1)

# On convertit l'image en nuances de gris pour travailler dessus
imgGreyscale = ImageManager.getGreyscale(imgWithColor)

# On charge un analyseur afin de récolter des données sur notre script
analyzer = PyramidAnalyzer("out/")
analyzer.originalPicture = imgWithColor
analyzer.greyscalePicture = imgGreyscale

# On applique l'algorithme
keypoints = ImageProcessor.findKeypoints(imgGreyscale, s, octave,
                                         verbose=DEBUG,
                                         show_images=SHOW_IMAGE,
                                         pyramid_analyzer=analyzer)
