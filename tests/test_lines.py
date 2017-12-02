# -*- coding: utf-8 -*-
"""
    Ce programme test la reaction du script face a divers structure simples
    Résultats : Un problème a été détécté sur la détéction d'extremum
"""


import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

Log.debug("Démarrage du test...")
Log_file.DEBUG_ACTIVATED = True

for i in range(1, 5):
    #  On charge l'image et on la redimensionne

    Log.debug("Chargement de l'image line" + str(i) + ".png")
    image = ImageManager.loadMatrix("images/" + "line" + str(i) + ".png")

    # On applique l'algorithme
    descriptors, analyzer = ImageProcessor.findKeypoints(image, 3, 2,
                                                         analysis=True,
                                                         analyzer_outdir="out_lines/" + str(i),
                                                         minimum_contrast=0.10,
                                                         r_courb_principal=10)

    analyzer.analyze()
