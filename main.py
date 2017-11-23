# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

IMAGES_PATH = "images/"
NAME_PICTURE = 'lena2.jpg'

ANALYSIS = True
DEBUG = True

s = 3
octave = 3

#  On charge l'image et on la redimensionne
Log_file.DEBUG_ACTIVATED = DEBUG
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
imgWithColor = ImageManager.loadMatrix(IMAGES_PATH + NAME_PICTURE)
Log.debug("Image chargée. Redimensionnement de l'image")
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

# On vérifie que le nombre d'octave n'st pas trop grand
octave_debug = min(int(np.log2(imgGreyscale.shape[0])), int(np.log2(imgGreyscale.shape[1])), octave)
if octave_debug != octave:
    Log.info("Le nombre d'octave a été changé à " + str(
        int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
    octave = octave_debug

# On applique l'algorithme
keypoints = ImageProcessor.findKeypoints(imgGreyscale, s, octave,
                                         verbose=DEBUG,
                                         pyramid_analyzer=analyzer)
