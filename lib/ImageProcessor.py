# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.analysis.PyramidAnalyzer import *


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def findKeypoints(image_original, s, nb_octaves, **kwargs):
        Log.debug("Doublement de la taille de l'image")
        image_doubled = ImageManager.getOctave(image_original, -1)

        # On convertit l'image en nuances de gris pour travailler dessus
        Log.debug("Conversion de l'image en niveaux de gris")
        image_greyscale = ImageManager.getGreyscale(image_doubled)

        # On charge un analyseur afin de récolter des données sur notre script
        if kwargs.get("analysis", False):
            Log.debug("Chargement de l'analyseur")
            pyramid_analyzer = PyramidAnalyzer(kwargs.get("analyzer_outdir", "out/"))
            pyramid_analyzer.originalPicture = image_original
            pyramid_analyzer.doubledImage = image_doubled
            pyramid_analyzer.greyscalePicture = image_greyscale
        else:
            pyramid_analyzer = None

        # On vérifie que le nombre d'octave n'est pas trop grand
        octave_debug = min(int(np.log2(image_greyscale.shape[0])), int(np.log2(image_greyscale.shape[1])), nb_octaves)
        if octave_debug != nb_octaves:
            Log.info("Le nombre d'octave a été changé à " + str(
                    int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
            octave = octave_debug

        # On construit la pyramide des gaussiennes
        Log.debug("Construction de la pyramide des DoG")
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image_greyscale, s, nb_octaves)
        points_cles = []

        if pyramid_analyzer:
            pyramid_analyzer.sigmas = sigmas
            pyramid_analyzer.dogPyramid = DoGs
            pyramid_analyzer.imagePyramid = octaves

        for i in range(len(octaves)):
            if kwargs.get("verbose", False):
                Log.debug("Debut de l'analyse de l'octave " + str(i + 1))

            points = ExtremaDetector.detectionPointsCles(
                    DoGs[i],
                    octaves[i],
                    sigmas,
                    kwargs.get("minimum_contrast", 0.20),
                    kwargs.get("r_courb_principal", 10),
                    1 / (2 ** i),
                    i,
                    pyramid_analyzer=pyramid_analyzer,
                    verbose=kwargs.get("verbose", False)
            )

            # On fait un rescale des points clés
            Log.debug("Rescale des points-clés sur l'image initiale", 1)
            octave_points = Utils.adaptKeypoints(points, i - 1)

            Log.debug("Remplacement des sigmas par leur valeur", 1)
            octave_points = Utils.adaptSigmas(octave_points, sigmas)

            if kwargs.get("verbose", False):
                Log.debug("Nombre de points pour l'octave " + str(i + 1) + " : " + str(len(octave_points)))

            # On ajoute les nouveaux points clés à la liste
            points_cles = Utils.concatenateKeyPoints(points_cles, octave_points)

        # Chargement de l'analyseur et lancement de l'analyse
        if pyramid_analyzer:
            pyramid_analyzer.keypoints = points_cles

        Log.debug("Nombre total de points clés : " + str(len(points_cles)))
        Log.debug("Calcul des descripteurs....")
        #descripteurs = ExtremaDetector.descriptionPointsCles(image_greyscale, points_cles)
        descripteurs = None

        if pyramid_analyzer:
            Log.debug("Enregistrement des descripteurs dans l'analyseur")
            pyramid_analyzer.descriptors = descripteurs

        return descripteurs
