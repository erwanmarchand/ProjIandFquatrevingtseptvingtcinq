# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.analysis.PyramidAnalyzer import *


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def prepareImage(image):
        Log.debug("Doublement de la taille de l'image")
        image_doubled = ImageManager.getOctave(image, -1)

        Log.debug("Conversion de l'image en niveaux de gris")
        image_greyscale = ImageManager.getGreyscale(image_doubled)

        return image_doubled, image_greyscale

    @staticmethod
    def checkNbOctave(image_travail, nb_octaves):
        octave_debug = min(int(np.log2(image_travail.shape[0])), int(np.log2(image_travail.shape[1])), nb_octaves)
        if octave_debug != nb_octaves:
            Log.warn("Le nombre d'octave a été changé à " + str(
                int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
            return octave_debug

        return nb_octaves

    @staticmethod
    def findKeypoints(image_original, s, nb_octaves, **kwargs):
        if image_original is None:
            raise Exception("Erreur : Aucune image envoyee")

        image_doubled, image_greyscale = ImageProcessor.prepareImage(image_original)
        nb_octaves = ImageProcessor.checkNbOctave(image_doubled, nb_octaves)

        Log.debug("Construction des pyramides des gaussiennes et des DoGs")
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image_greyscale, s, nb_octaves)
        points_cles = []

        #  On applique une détéction de points clés sur chaque octave
        for i in range(len(octaves)):
            Log.debug("BEGIN -------------------------------- " + str(i + 1))
            Log.debug("Debut de l'analyse de l'octave " + str(i + 1))

            points = ExtremaDetector.detectionPointsCles(
                DoGs[i],
                octaves[i],
                sigmas,
                kwargs.get("minimum_contrast", 0.20),
                kwargs.get("r_courb_principal", 10),
                1 / (2 ** i),
                i,
                **kwargs
            )

            # On fait un rescale des points clés
            Log.debug("Rescale des points-clés sur l'image initiale", 1)
            octave_points = Utils.adaptKeypoints(points, i - 1)

            Log.debug("Nombre de points pour l'octave " + str(i + 1) + " : " + str(len(octave_points)))

            # On ajoute les nouveaux points clés à la liste
            points_cles = Utils.concatenateKeyPoints(points_cles, octave_points)

            Log.debug("END -------------------------------- " + str(i + 1))

        Log.debug("Remplacement des sigmas par leur valeur", 1)
        points_cles = Utils.adaptSigmas(points_cles, sigmas)

        Log.debug("Nombre total de points clés : " + str(len(points_cles)))

        Log.debug("Calcul des descripteurs....")
        descriptors = ExtremaDetector.descriptionPointsCles(image_greyscale, points_cles)

        # On s'occupe d'un eventuel analyseur
        ImageProcessor.fillAnalyzer(image_original,
                                    image_doubled,
                                    image_greyscale,
                                    DoGs,
                                    octaves,
                                    sigmas,
                                    points_cles,
                                    descriptors,
                                    **kwargs)

        return descriptors

    @staticmethod
    def fillAnalyzer(image_original, image_doubled, image_greyscale, DoGs, octaves, sigmas, key_points, descriptors,
                     **kwargs):
        pyramid_analyzer = kwargs.get("pyramid_analyzer", None)

        if pyramid_analyzer is not None:
            Log.debug("Remplissage de l'analyseur")
            pyramid_analyzer.originalPicture = copy.deepcopy(image_original)
            pyramid_analyzer.doubledPicture = copy.deepcopy(image_doubled)
            pyramid_analyzer.greyscaleDoubledPicture = copy.deepcopy(image_greyscale)

            pyramid_analyzer.sigmas = copy.deepcopy(sigmas)

            pyramid_analyzer.dogPyramid = copy.deepcopy(DoGs)
            pyramid_analyzer.imagePyramid = copy.deepcopy(octaves)

            pyramid_analyzer.key_points = copy.deepcopy(key_points)

            Log.debug("Enregistrement des descripteurs dans l'analyseur")
            pyramid_analyzer.descriptors = copy.deepcopy(descriptors)