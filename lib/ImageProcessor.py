# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.Utils import *
from lib.analysis.PyramidAnalyzer import *


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def findKeypoints(image, s, nb_octaves, **kwargs):
        # On construit la pyramide des gaussiennes
        Log.debug("Construction de la pyramide des DoG")
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image, s, nb_octaves, **kwargs)
        points_cles = []

        # Chargement de l'analyseur
        pyramidAnalyzer = kwargs.get("pyramid_analyzer", None)

        if pyramidAnalyzer:
            pyramidAnalyzer.sigmas = sigmas
            pyramidAnalyzer.dogPyramid = DoGs
            pyramidAnalyzer.imagePyramid = octaves

        for i in range(len(octaves)):
            if kwargs.get("verbose", False):
                Log.debug("Debut de l'analyse de l'octave " + str(i + 1))

            points = ExtremaDetector.detectionPointsCles(
                DoGs[i],
                octaves[i],
                sigmas,
                0.20,
                7.5,
                1 / (2 ** i),
                i,
                pyramid_analyzer=pyramidAnalyzer,
                verbose=kwargs.get("verbose", False)
            )

            # On fait un rescale des points clés
            Log.debug("Rescale des points clés", 1)
            octavePoints = Utils.adaptKeypoints(points, i)

            Log.debug("Remplacement des sigmas par leur valeurs", 1)
            octavePoints = Utils.adaptSigmas(octavePoints, sigmas)

            if kwargs.get("verbose", False):
                Log.debug("Nombre de points pour l'octave " + str(i + 1) + " : " + str(len(octavePoints)))

            # On ajoute les nouveaux points clés
            points_cles = Utils.concatenateKeyPoints(octavePoints, points_cles)

        # Chargement de l'analyseur et lancement de l'analyse
        if pyramidAnalyzer:
            pyramidAnalyzer.keypoints = points_cles

        Log.debug("Nombre total de points clés : " + str(len(points_cles)))
        Log.debug("Calcul des descripteurs....")
        descripteurs = ExtremaDetector.descriptionPointsCles(image, points_cles)

        return descripteurs
