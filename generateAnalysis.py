# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

IMAGES_PATH = "images/"
ANALYSIS = True
DEBUG = True
S = 3
Log_file.DEBUG_ACTIVATED = DEBUG

NAME_PICTURE = 'gauche.jpg'
# Image de gauche, 4 octaves

#  On charge l'image
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
image = ImageManager.loadMatrix(IMAGES_PATH + NAME_PICTURE)

# On applique l'algorithme
analyzer = PyramidAnalyzer("out_gauche_4/")
ImageProcessor.findKeypoints(image, S, 4,
                             pyramid_analyzer=analyzer,
                             minimum_contrast=0.03,
                             r_courb_principal=10)

analyzer.analyze()

# Image de gauche, 1 octaves
#  On charge l'image
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
image = ImageManager.loadMatrix(IMAGES_PATH + NAME_PICTURE)

# On applique l'algorithme
analyzer = PyramidAnalyzer("out_gauche_1/")
ImageProcessor.findKeypoints(image, S, 1,
                             pyramid_analyzer=analyzer,
                             minimum_contrast=0.03,
                             r_courb_principal=10)

analyzer.analyze()

# --------------------------------------------------- #

NAME_PICTURE = 'droite.jpg'
# Image de droite, 4 octaves
#  On charge l'image
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
image = ImageManager.loadMatrix(IMAGES_PATH + NAME_PICTURE)

# On applique l'algorithme
analyzer = PyramidAnalyzer("out_droite_4/")
ImageProcessor.findKeypoints(image, S, 4,
                             pyramid_analyzer=analyzer,
                             minimum_contrast=0.03,
                             r_courb_principal=10)

analyzer.analyze()

# Image de droite, 1 octaves
#  On charge l'image
Log.debug("Démarrage...")
Log.debug("Chargement de l'image")
image = ImageManager.loadMatrix(IMAGES_PATH + NAME_PICTURE)

# On applique l'algorithme
analyzer = PyramidAnalyzer("out/out_droite_1/")
ImageProcessor.findKeypoints(image, S, 1,
                             pyramid_analyzer=analyzer,
                             minimum_contrast=0.03,
                             r_courb_principal=10)

analyzer.analyze()
