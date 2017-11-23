# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.Panorama import *
from lib.analysis.PyramidAnalyzer import *
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

# On crée le panorama

(SIFTPointsLeft,SIFTPointsRight) = Panorama.getSIFTPoints(imgGaucheWithColor, imgDroiteWithColor)

distEuc = Panorama.distanceInterPoints(SIFTPointsLeft,SIFTPointsRight)

test = 3
