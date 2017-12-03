# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.Panorama import *
from lib.analysis.PanoramaAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

IMAGES_PATH = "images/"
#NAME_IMAGE_GAUCHE = 'lena2left.jpg'
#NAME_IMAGE_DROITE = 'lena2right.jpg'
NAME_IMAGE_GAUCHE = 'gauche.jpg'
NAME_IMAGE_DROITE = 'droite.jpg'

ANALYSIS = True
ANALYSE_EACH_IMAGE = False
DEBUG = True

#  On charge les images et on les redimensionne
Log_file.DEBUG_ACTIVATED = DEBUG
Log.debug("Démarrage...")
Log.debug("Chargement des images")
image_gauche = ImageManager.loadMatrix(IMAGES_PATH + NAME_IMAGE_GAUCHE)
image_droite = ImageManager.loadMatrix(IMAGES_PATH + NAME_IMAGE_DROITE)

# On charge un analyseur afin de récolter des données sur notre script
if ANALYSIS:
    Log.debug("Chargement de l'analyseur")
    analyzer = PanoramaAnalyzer("out_panorama/")
    analyzer.originalLeftPicture = image_gauche
    analyzer.originalRightPicture = image_droite
else:
    analyzer = None


# On crée le panorama
minValues = Panorama.getFriendlyCouples(image_gauche, image_droite, 10,
                                        panorama_analyzer=analyzer,
                                        analyse_each_image=ANALYSE_EACH_IMAGE)

A = Panorama.getMatriceA(minValues)

Hnorm = Panorama.getTransformMatrix(A)

finalPicture = Panorama.generatePanorama(image_gauche,image_droite,Hnorm, panorama_analyzer=analyzer)

analyzer.analyze()
