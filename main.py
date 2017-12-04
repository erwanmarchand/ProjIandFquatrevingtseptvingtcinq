# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

IMAGES_PATH = "images/"
NAME_PICTURE = 'gauche.jpg'

ANALYSIS = True
DEBUG = True

S = 3
NB_OCTAVE = 1

#  On charge l'image et on la redimensionne
Log_file.DEBUG_ACTIVATED = DEBUG
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
image = ImageManager.loadMatrix(IMAGES_PATH + NAME_PICTURE)

# On applique l'algorithme
analyzer = PyramidAnalyzer("out_gauche/")
descriptors = ImageProcessor.findKeypoints(image, S, NB_OCTAVE,
                                           pyramid_analyzer=analyzer,
                                           minimum_contrast=0.03,
                                           r_courb_principal=10)

analyzer.analyze()
