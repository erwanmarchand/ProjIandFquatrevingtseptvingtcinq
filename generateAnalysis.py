# -*- coding: utf-8 -*-
"""
    Ce fichier permet de générer l'analyse des points clés de plusieurs fichiers, avec plusieurs paramètres
"""

from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import lib.Utils as Utils_file

# Configuration générale
IMAGES_PATH = "images/"
OUT_DIR = "out/"
Log_file.DEBUG_ACTIVATED = True  # False si vous voulez que les logs soit désactivés
Utils_file.ACTIVATE_LOAD_BAR = True  # False si vous voulez que les load bar soit désactivés (sous windows notamnent)

# Paramètres de tests
PICTURES = ["gauche.jpg", "droite.jpg", "notre-dame.png", "lena.jpg"]
CONSTRAST_ARGS = [0.01, 0.02, 0.03]
HESSIAN_ARGS = [5, 7, 10]
NB_OCTAVES = [1, 4]

#  Execution
for p in PICTURES:
    for n in NB_OCTAVES:
        for h in HESSIAN_ARGS:
            for c in CONSTRAST_ARGS:
                print("#########################")
                p_name = p.split('.')[0]

                Log.debug("Chargement de l'image " + p_name)
                image = ImageManager.loadMatrix(IMAGES_PATH + p)

                # On applique l'algorithme
                analyzer = PyramidAnalyzer(OUT_DIR + p_name + "_" + str(n) + "_" + str(c) + "_" + str(h) + "/")
                ImageProcessor.findKeypoints(image, 3, n,
                                             pyramid_analyzer=analyzer,
                                             minimum_contrast=c,
                                             r_courb_principal=h)

                analyzer.analyze()
