# -*- coding: utf-8 -*-
"""
    Ce programme test la reaction de la Hessienne face a une ligne droite
    Résultats : Il faut rajouter une valeur absolue pour un résultat satisfaisant
"""

import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import copy

Log.debug("Démarrage du test...")
Log_file.DEBUG_ACTIVATED = True

#  On charge l'image et on la redimensionne
image = ImageManager.loadMatrix("images/" + "figure.png")

# On applique l'algorithme
for hp in range(5, 25, 2)[::-1]:
    analyzer = PyramidAnalyzer("out_line1/" + str(hp))
    descriptors, analyzer = ImageProcessor.findKeypoints(copy.deepcopy(image), 3, 2,
                                                         pyramid_analyzer=analyzer,
                                                         minimum_contrast=0.10,
                                                         r_courb_principal=hp)
    analyzer.SAVE_DPI = 400
    analyzer.analyze()
