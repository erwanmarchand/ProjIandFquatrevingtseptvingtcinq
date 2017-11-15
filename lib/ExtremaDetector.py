# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from lib.ImageManager import *
from lib.debug.Log import *


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

        return doG, pyramid, sigmas

    @staticmethod
    def detectionPointsCles(DoGs, octaves, sigmas, seuil_contraste, r_courb_principale, resolution_octave):
        # Quelques vérifications d'usage afin de garantir le bon déroulement de la méthode
        if len(DoGs) == 0:
            raise "Erreur : Aucune image DoGs fournie en parametre"

        # Execution
        width, height = ImageManager.getDimensions(DoGs[0])

        def _detectionExtremums():
            extremums = []

            for i in range(len(DoGs[1:-1])):
                # On fait une boucle sur l'ensemble des pixels, bords exclus, afin de ne pas avoir a faire du cas par cas
                for x in range(1, height - 1):
                    for y in range(1, width - 1):
                        neighboors = []

                        neighboors += [DoGs[i - 1][x - 1:x + 1, y - 1:y + 1]]
                        neighboors += [DoGs[i][x - 1:x + 1, y - 1:y + 1]]
                        neighboors += [DoGs[i + 1][x - 1:x + 1, y - 1:y + 1]]

                        # Si le point est effectivement le maximum de la region, c'est un point candidat
                        if DoGs[i][x, y] == np.max(neighboors):
                            extremums.append((x, y, i))

            return extremums


        def _filtrerPointsContraste(candidats):
            realPoints = []

            for c in candidats:
                (x, y, i) = c
                if DoGs[i][x, y] > seuil_contraste:
                    realPoints.append(c)

            return realPoints

        def _filtrerPointsArete(candidats):
            realPoints = candidats


            # On calcul la Hessienne

            return realPoints

        candidats = _detectionExtremums()
        ## BONUS EVENTUEL ICI
        candidats = _filtrerPointsContraste(candidats)
        candidats = _filtrerPointsArete(candidats)

        return candidats

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
