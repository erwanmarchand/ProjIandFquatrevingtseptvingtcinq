# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.Panorama import *
from lib.analysis.PanoramaAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

IMAGES_PATH = "images/"
NAME_IMAGE_GAUCHE = 'gauche.jpg'
NAME_IMAGE_DROITE = 'droite.jpg'

ANALYSIS = True
DEBUG = True

#  On charge les images et on les redimensionne
Log_file.DEBUG_ACTIVATED = DEBUG
Log.debug("Démarrage...")
Log.debug("Chargement des images")
imgGaucheWithColor = ImageManager.loadMatrix(IMAGES_PATH + NAME_IMAGE_GAUCHE)
imgDroiteWithColor = ImageManager.loadMatrix(IMAGES_PATH + NAME_IMAGE_DROITE)
Log.debug("Images chargées. Redimensionnement des images")
imgGaucheWithColor = ImageManager.getOctave(imgGaucheWithColor, -1)
imgDroiteWithColor = ImageManager.getOctave(imgDroiteWithColor, -1)

# On charge un analyseur afin de récolter des données sur notre script
if ANALYSIS:
    Log.debug("Chargement de l'analyseur")
    analyzer = PanoramaAnalyzer("out/")
    analyzer.originalLeftPicture = imgGaucheWithColor
    analyzer.originalRightPicture = imgDroiteWithColor
else:
    analyzer = None

# On crée le panorama

minValues = Panorama.getFriendlyCouples(imgGaucheWithColor, imgDroiteWithColor, 4, verbose=DEBUG, panorama_analyzer=analyzer)

#analyzer.analyze()