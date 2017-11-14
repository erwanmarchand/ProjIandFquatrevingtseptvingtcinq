# -*- coding: utf-8 -*-
from lib.Image import *
import matplotlib.pyplot as plt


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
            octave_original = Image.getOctave(image, octave)

            for k in range(nb_element):
                pyramid[octave].append(Image.applyGaussianFilter(octave_original, sigmas[k]))

        if verbose:
            ExtremaDetector.showPyramid(pyramid, sigmas, "Pyramide des images filtrées par un filtre gaussien")

        # Make difference
        doG = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            for k in range(nb_element - 1):
                doG[octave].append(Image.makeDifference(pyramid[octave][k + 1], pyramid[octave][k]))

        if verbose:
            ExtremaDetector.showPyramid(doG, sigmas, "Pyramide des DoGs")

        return doG, sigmas

    @staticmethod
    def detectionPointsCles(DoGs, octaves, sigmas, seuil_contraste, r_courb_principale, resolution_octave):
        pass

    @staticmethod
    def showPyramid(pyramid, sigmas, title = ""):
        """
        Open a window and show a Pyramid
        :param pyramid: The pyramid we want to watch
        :param sigmas: The differents sigmas of the pyramid
        :return: 
        """
        nb_octave, nb_per_row = len(pyramid), len(pyramid[0])

        # Show pyramid
        for o in range(nb_octave):
            for k in range(nb_per_row):
                plt.subplot(nb_octave, nb_per_row, (k + 1) + o * nb_per_row)
                plt.title(str(o + 1) + " || " + str(round(sigmas[k], 4)))
                plt.imshow(pyramid[o][k], cmap=None)

        plt.show()
