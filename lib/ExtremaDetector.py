# -*- coding: utf-8 -*-
from lib.ImageManager import *
import matplotlib.pyplot as plt
from lib.Log import *


class ExtremaDetector:
    @staticmethod
    def differenceDeGaussienne(image, s, nb_octave, **kwargs):
        """
        Génère la pyramide des DoG d'une image
        :param image: L'image originale
        :param s: Le facteur s
        :param nb_octave: Le nombre d'octave sur lesquelles on veut travailler
        :return: (DoGs, sigmas)
        """
        if image is None:
            raise Exception("Erreur : Aucune image envoyee")

        nb_element = s + 3
        verbose = kwargs.get('verbose', False)

        # Construction de la pyramide de gaussiennes
        sigmas = [1.6 * 2 ** (float(k) / float(s)) for k in range(nb_element)]
        pyramid = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            octave_original = ImageManager.getOctave(image, octave)

            for k in range(nb_element):
                pyramid[octave].append(ImageManager.applyGaussianFilter(octave_original, sigmas[k]))

        if verbose:
            ExtremaDetector.showPyramid(pyramid, sigmas, title="Pyramide des images filtrees par un filtre gaussien")

        # Make difference
        doG = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            for k in range(nb_element - 1):
                doG[octave].append(ImageManager.makeDifference(pyramid[octave][k + 1], pyramid[octave][k]))

        if verbose:
            ExtremaDetector.showPyramid(doG, sigmas, title="Pyramide des DoGs")

        return doG, sigmas

    @staticmethod
    def detectionPointsCles(DoGs, octaves, sigmas, seuil_contraste, r_courb_principale, resolution_octave):
        def _detectionExtremums():
            extremums = []

            return extremums

        def _corrigerPositionExtremum(extremum):
            return extremum

        def _filtrerPointsCandidats(candidats):
            pass

        candidats = _detectionExtremums()
        for i, ext in candidats:
            candidats[i] = _corrigerPositionExtremum(ext)
        pointsCles = _filtrerPointsCandidats(candidats)

        return pointsCles


    @staticmethod
    def showPyramid(pyramid, sigmas, **kwargs):
        """
        Open a window and show a Pyramid
        :param pyramid: The pyramid we want to watch
        :param sigmas: The differents sigmas of the pyramid
        """
        nb_octave, nb_per_row = len(pyramid), len(pyramid[0])

        # Show pyramid
        Log.info("Affichage d'une pyramide : " + kwargs.get("title", "noname"))
        for o in range(nb_octave):
            for k in range(nb_per_row):
                plt.subplot(nb_octave, nb_per_row, (k + 1) + o * nb_per_row)
                plt.title(str(o + 1) + " || " + str(round(sigmas[k], 4)))
                plt.imshow(pyramid[o][k], cmap=kwargs.get("cmap", 'gray'))

        plt.show()
