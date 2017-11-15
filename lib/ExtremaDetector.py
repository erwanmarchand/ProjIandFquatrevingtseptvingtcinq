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

        if len(DoGs) < 3:
            raise "Erreur : Pas assez d'images DoGs fournies en paramètre (3 minimum)"

        # Execution
        height, width = ImageManager.getDimensions(DoGs[0])

        # Génération des DoGs normalisés (valeur en 0 et 1)
        DoGsNormalized = []
        for DoG in DoGs:
            DoGsNormalized.append(ImageManager.normalizeImage(DoG))

        def _detectionExtremums():
            extremums = []

            for i in range(len(DoGs[1:-1])):
                # On fait une boucle sur l'ensemble des pixels, bords exclus afin de ne pas avoir a faire du cas par cas
                for x in range(1, height - 1):
                    for y in range(1, width - 1):
                        neighbours = []

                        neighbours += [DoGs[i - 1][x - 1:x + 1, y - 1:y + 1]]
                        neighbours += [DoGs[i][x - 1:x + 1, y - 1:y + 1]]
                        neighbours += [DoGs[i + 1][x - 1:x + 1, y - 1:y + 1]]

                        # Si le point est effectivement le maximum de la region, c'est un point candidat
                        if DoGs[i][x, y] == np.max(neighbours):
                            extremums.append((x, y, i))

            return extremums

        def _filtrerPointsContraste(candidats):
            realPoints = []

            for c in candidats:
                (x, y, i) = c
                if DoGsNormalized[i][x, y] > seuil_contraste:
                    realPoints.append(c)

            return realPoints

        def _filtrerPointsArete(candidats):
            realPoints = candidats

            # On calcul la Hessienne

            return realPoints

        def _assignOrientation(candidats):
            points = []

            VOISINAGE_COTE = 5  # Pour n, on va faire un carre de x-n:x+n, y-n:y+n
            # On creer le slice de l'histogramme
            H_slice = np.linspace(0, 2 * np.pi, 36 + 1) # 37 valeurs, donc 36 intervalles

            for c in candidats:
                (x, y, i) = c
                H = np.zeros(36)

                # Selection des cotés du voisinage en faisant attention aux bords
                xMax, yMax, xMin, yMin = min(height - 1, x + VOISINAGE_COTE), \
                                         min(width - 1, y + VOISINAGE_COTE), \
                                         max(1, x - VOISINAGE_COTE), \
                                         max(1, y - VOISINAGE_COTE)

                base = octaves[i][xMin:xMax, yMin:yMax]
                g, d, b, h = octaves[i][(xMin - 1):(xMax - 1), yMin:yMax], \
                             octaves[i][(xMin + 1):(xMax + 1), yMin:yMax], \
                             octaves[i][xMin:xMax, (yMin - 1):(yMax - 1)], \
                             octaves[i][xMin:xMax, (yMin + 1):(yMax + 1)]

                # Calcul des amplitude des gradients et de l'orientation
                M = np.sqrt(np.power(d - g, 2) + np.power(b - h, 2))
                A = np.arctan((b - h) / (d - g))

                # Analyse des résultats, on applatit le slice de la matrice
                Ms, As = M.flat, A.flat

                for angle in As:
                    for si in range(36):
                        if angle <= H_slice[si + 1]:
                            H[si] += 1


            return candidats

        candidats = _detectionExtremums()
        ## BONUS EVENTUEL ICI
        candidats = _filtrerPointsContraste(candidats)
        candidats = _filtrerPointsArete(candidats)
        candidats = _assignOrientation(candidats)

        return candidats

    @staticmethod
    def showPyramid(pyramid, sigmas, **kwargs):
        """
        Open a window and show a Pyramid
        :param pyramid: The pyramid we want to watch
        :param sigmas: The different sigmas of the pyramid
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
