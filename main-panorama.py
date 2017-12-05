# -*- coding: utf-8 -*-
"""
    Ce fichier permet de générer un panorama à partir de deux iamges ainsi qu'une analyse de la génération
"""
from lib.Panorama import *
from lib.analysis.PanoramaAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file

# CONFIGURATION
# ---------------------------------- #
IMAGES_PATH = "images/"
NAME_IMAGE_GAUCHE = 'gauche.jpg'
NAME_IMAGE_DROITE = 'droite.jpg'

ANALYSIS = True
ANALYSE_EACH_IMAGE = False
DEBUG = True

# EXECUTION
# ---------------------------------- #
#  On charge les images
Log.debug("Démarrage...")
Log.debug("Chargement des images")

Log_file.DEBUG_ACTIVATED = DEBUG
image_gauche = ImageManager.loadMatrix(IMAGES_PATH + NAME_IMAGE_GAUCHE)
image_droite = ImageManager.loadMatrix(IMAGES_PATH + NAME_IMAGE_DROITE)

# On charge un analyseur afin de récolter des données sur notre script
if ANALYSIS:
    Log.debug("Chargement de l'analyseur")
    analyzer = PanoramaAnalyzer("out/panorama/")
    analyzer.originalLeftPicture = image_gauche
    analyzer.originalRightPicture = image_droite
else:
    analyzer = None

# On crée le panorama
finalPicture = Panorama.generatePanorama(image_gauche, image_droite,
                                         panorama_analyzer=analyzer,
                                         analyse_each_image=ANALYSE_EACH_IMAGE)

if ANALYSIS:
    analyzer.analyze()
