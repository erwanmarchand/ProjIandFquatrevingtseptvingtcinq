# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.debug.Log import *
from lib.analysis.PyramidAnalyzer import *
from lib.Utils import *

import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def findKeypoints(image, s, nb_octaves, **kwargs):
        # On construit la pyramide des gaussiennes
        Log.debug("Construction de la pyramide des DoG")
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image, s, nb_octaves, **kwargs)
        realPoints = []

        # Chargement de l'analyseur
        pyramidAnalyzer = kwargs.get("pyramid_analyzer", None)

        if pyramidAnalyzer:
            pyramidAnalyzer.sigmas = sigmas
            pyramidAnalyzer.dogPyramid = DoGs
            pyramidAnalyzer.imagePyramid = octaves

        for i in range(len(octaves)):
            if kwargs.get("verbose", False):
                Log.debug("Debut de l'analyse de l'octave " + str(i))

            points = ExtremaDetector.detectionPointsCles(
                    DoGs[i],
                    octaves[i],
                    sigmas,
                    0.08,
                    10,
                    1 / (2 ** i),
                    i,
                    pyramid_analyzer=pyramidAnalyzer,
                    verbose=kwargs.get("verbose", False)
            )

            # On fait un rescale des points clés
            octavePoints = Utils.adaptKeypoints(points, i)
            octavePoints = Utils.adaptSigmas(octavePoints, sigmas)

            for elt in octavePoints:
                print(elt)
            exit(0)

            if kwargs.get("verbose", False):
                Log.debug("Nombre de points pour l'octave " + str(i) + " : " + str(len(octavePoints)))

            # On ajoute les nouveaux points clés
            realPoints = Utils.concatenateKeyPoints(octavePoints, realPoints)

        # Chargement de l'analyseur et lancement de l'analyse
        if pyramidAnalyzer:
            pyramidAnalyzer.keypoints = realPoints
            pyramidAnalyzer.analyze()

        return realPoints
