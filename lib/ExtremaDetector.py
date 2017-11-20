# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import copy

from lib.ImageManager import *
from lib.debug.Log import *
from lib.analysis.Utils import *
from lib.analysis.OctaveAnalyzer import *


class ExtremaDetector:
    def __init__(self):
        pass

    @staticmethod
    def differenceDeGaussienne(image, s, nb_octave, **kwargs):
        """
        Génère la pyramide des DoG d'une image
        :param image:       L'image originale
        :param s:           Le facteur s
        :param nb_octave:   Le nombre d'octave sur lesquelles on veut travailler
        :return:            (DoGs, pyramide des octaves, liste des sigmas)
        """
        if image is None:
            raise Exception("Erreur : Aucune image envoyee")

        nb_element = s + 3
        verbose, show_images = kwargs.get('verbose', False), kwargs.get('show_images', False)

        # Construction de la pyramide de gaussiennes
        sigmas = [1.6 * 2 ** (float(k) / float(s)) for k in range(nb_element)]
        pyramid = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            octave_original = ImageManager.getOctave(image, octave)

            for k in range(nb_element):
                pyramid[octave].append(ImageManager.applyGaussianFilter(octave_original, sigmas[k]))

        if show_images:
            Utils.showPyramid(pyramid, sigmas, title="Pyramide des images filtrees par un filtre gaussien")

        # Make difference
        doG = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            for k in range(nb_element - 1):
                doG[octave].append(ImageManager.makeDifference(pyramid[octave][k + 1], pyramid[octave][k]))

        if show_images:
            Utils.showPyramid(doG, sigmas, title="Pyramide des DoGs")

        return doG, pyramid, sigmas

    @staticmethod
    def detectionPointsCles(DoGs, octaves, sigmas, seuil_contraste, r_courb_principale, resolution_octave, octave_nb,
                            **kwargs):
        """
        Detecte les points clés d'une image
        :param DoGs:                Liste des DoGs pour une octave
        :param octaves:             Liste des images d'une octave
        :param sigmas:              Liste des sigmas (provenant de la convolution par une gaussienne)
        :param seuil_contraste:     Seuil de contraste
        :param r_courb_principale:  Rayon de courbure principal
        :param resolution_octave:   Résolution de l'octave
        :param octave_nb:           Numéro de l'octave
        :return:                    Liste des points clés
        """

        # Quelques vérifications d'usage afin de garantir le bon déroulement de la méthode
        if len(DoGs) == 0:
            raise ("Erreur : Aucune image DoGs fournie en parametre")

        if len(DoGs) < 3:
            raise ("Erreur : Pas assez d'images DoGs fournies en paramètre (3 minimum)")

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

                        neighbours += [DoGs[i - 1][x - 1:x + 2, y - 1:y + 2]]
                        neighbours += [DoGs[i][x - 1:x + 2, y - 1:y + 2]]
                        neighbours += [DoGs[i + 1][x - 1:x + 2, y - 1:y + 2]]

                        neighbours = np.array(neighbours).flat

                        # Si le point est effectivement le maximum de la region, c'est un point candidat
                        if DoGs[i][x, y] == np.max(neighbours) or DoGs[i][x, y] == np.min(neighbours):
                            extremums.append((x, y, i))

            return extremums

        def _filtrerPointsContraste(candidats):
            realPoints = []

            for c in candidats:
                (x, y, i) = c
                if abs(DoGsNormalized[i][x, y]) > seuil_contraste:
                    realPoints.append(c)

            return realPoints

        def _filtrerPointsArete(candidats):
            realPoints = []

            # On calcul la Hessienne
            for c in candidats:
                (x, y, i) = c
                hessianxx = DoGs[i][x + 1, y] + DoGs[i][x - 1, y] - (2 * DoGs[i][x, y])
                hessianxy = DoGs[i][x + 1, y + 1] - DoGs[i][x, y + 1] - DoGs[i][
                    x + 1, y] + DoGs[i][x, y]
                hessianyy = DoGs[i][x, y + 1] + DoGs[i][x, y - 1] - (2 * DoGs[i][x, y])

                Tr = hessianxx + hessianyy
                Det = (hessianxx * hessianyy) - (hessianxy ** 2)

                R = (Tr ** 2) / Det
                rapport = ((r_courb_principale + 1) ^ 2) / r_courb_principale

                if R < rapport:
                    realPoints.append(c)

            return realPoints

        def _assignOrientation(candidats):
            realPoints = []
            VOISINAGE_COTE = 7  # Pour n, on va chercher dans un voisinage de x-n:x+n, y-n:y+n pixels (n+1)**2
            # On creer le slice de l'histogramme
            H_slice = np.linspace(0, 2 * np.pi, 36 + 1)  # 37 valeurs, donc 36 intervalles

            for c in candidats:
                (x, y, i) = c
                H = np.zeros(36)

                # Selection des points du voisinage en faisant attention aux bords
                xMax, yMax, xMin, yMin = min(height - 1, x + VOISINAGE_COTE), \
                                         min(width - 1, y + VOISINAGE_COTE), \
                                         max(1, x - VOISINAGE_COTE), \
                                         max(1, y - VOISINAGE_COTE)

                g, d, b, h = octaves[i][(xMin - 1):(xMax - 1), yMin:yMax], \
                             octaves[i][(xMin + 1):(xMax + 1), yMin:yMax], \
                             octaves[i][xMin:xMax, (yMin - 1):(yMax - 1)], \
                             octaves[i][xMin:xMax, (yMin + 1):(yMax + 1)]

                # Calcul des amplitude des gradients et de l'orientation
                M = np.sqrt(np.power(d - g, 2) + np.power(b - h, 2))
                A = np.arctan((b - h) / (d - g))

                # Analyse des résultats, on applatit le carré de matrice
                Ms, As = M.flat, A.flat

                for k, angle in enumerate(As):
                    for si in range(36):
                        if angle <= H_slice[si + 1]:
                            H[si] += Ms[k]
                            break

                # On selectionne les angles ayant plus de 80% de la valeur maximale
                mH, angles = np.max(H), []
                for k, a in enumerate(H):
                    if H[k] / mH >= 0.80:
                        angles.append(H_slice[k + 1])

                for a in angles:
                    realPoints.append((x, y, i, a))

            return realPoints

        # On initialise la tableau d'analyse pour l'analyseur
        pyramid_analyzer = kwargs.get("pyramid_analyzer", None)
        analyseur = OctaveAnalyzer(octave_nb) if pyramid_analyzer else None
        elements = {} if analyseur else None

        # Detection des extremums
        candidats = _detectionExtremums()
        elements["kp_after_extremum_detection"] = copy.deepcopy(candidats) if analyseur else []

        ## BONUS EVENTUEL ICI

        # Filtrage des points de faible contrastes
        Log.info("\t" + str(len(candidats)) + " points avant le filtrage par contraste")
        candidats = _filtrerPointsContraste(candidats)
        elements["kp_after_contrast_limitation"] = copy.deepcopy(candidats) if analyseur else []

        # Filtrage des points sur les arêtes
        Log.info("\t" + str(len(candidats)) + " points avant le filtrage des arêtes")
        candidats = _filtrerPointsArete(candidats)
        elements["kp_after_hessian_filter"] = copy.deepcopy(candidats) if analyseur else []

        # Assignation de l'orientation des points
        Log.info("\t" + str(len(candidats)) + " points avant l'assignation d'orientation")
        candidats = _assignOrientation(candidats)
        elements["kp_after_orientation_assignation"] = copy.deepcopy(candidats) if analyseur else []

        # Packaging des points clés et des outils d'analyse
        if analyseur:
            analyseur.finalKeypoints = copy.deepcopy(candidats)
            analyseur.elements = elements
            pyramid_analyzer.addOctaveAnalyzer(analyseur)

        return candidats
