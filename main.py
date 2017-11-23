# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *
import cv2

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"
ANALYSIS = True
DEBUG = True
NAME_PICTURE = 'lena2.jpg'

s = 3
octave = 5



#  On charge l'image et on la redimensionne
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
imgWithColor = ImageManager.loadMatrix(ORIGINAL_IMAGES_PATH + NAME_PICTURE)
Log.debug("Redimensionnement de l'image")
imgWithColor = ImageManager.getOctave(imgWithColor, -1)

# On convertit l'image en nuances de gris pour travailler dessus
Log.debug("Conversion de l'image en niveaux de gris")
imgGreyscale = ImageManager.getGreyscale(imgWithColor)

# On charge un analyseur afin de récolter des données sur notre script
if ANALYSIS:
    Log.debug("Chargement de l'analyseur")
    analyzer = PyramidAnalyzer("out/")
    analyzer.originalPicture = imgWithColor
    analyzer.greyscalePicture = imgGreyscale
else:
    analyzer = None

# On applique l'algorithme
keypoints = ImageProcessor.findKeypoints(imgGreyscale, s, octave,
                                         verbose=DEBUG,
                                         pyramid_analyzer=analyzer)
