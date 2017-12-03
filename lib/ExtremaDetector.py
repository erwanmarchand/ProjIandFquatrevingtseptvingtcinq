# -*- coding: utf-8 -*-
import copy

from lib.ImageManager import *
from lib.analysis.OctaveAnalyzer import *
from lib.Utils import *
from lib.debug.Log import *


class ExtremaDetector:
    def __init__(self):
        pass

    @staticmethod
    def differenceDeGaussienne(image, s, nb_octave, **kwargs):
        """
        Génère la pyramide des DoGs d'une image
        :param image:       L'image originale
        :param s:           Le facteur s
        :param nb_octave:   Le nombre d'octave à générer
        :return:            (pyramide des DoGs, pyramide des octaves, liste des sigmas)
        """

        nb_element = s + 3

        # Construction de la pyramide de gaussiennes
        Log.debug("Génération de la pyramide de Gaussiennes", 1)
        sigmas = [1.6 * (2 ** (float(k) / float(s))) for k in range(nb_element)]
        pyramid = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            if octave == 0:
                octave_original = image
            else:
                #  On prend la s + 1 eme image (la numéro s dans le tableau)
                octave_original = ImageManager.divideSizeBy2(pyramid[octave - 1][s])

            for k in range(nb_element):
                pyramid[octave].append(ImageManager.applyGaussianFilter(octave_original, sigmas[k]))

        # Generation de la pyramide des différences
        Log.debug("Génération de la pyramide des différences de Gaussiennes", 1)
        DoGs = [[] for k in range(nb_octave)]

        for octave in range(nb_octave):
            for k in range(nb_element - 1):
                DoGs[octave].append(pyramid[octave][k + 1] - pyramid[octave][k])

        return DoGs, pyramid, sigmas

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

        # Quelques vérifications d'usages afin de garantir le bon déroulement de la méthode
        if len(DoGs) == 0:
            raise ("Erreur : Aucune image DoGs fournie en parametre")

        if len(DoGs) < 3:
            raise ("Erreur : Pas assez d'images DoGs fournies en paramètre (3 minimum)")

        # Execution
        height, width = ImageManager.getDimensions(DoGs[0])

        # Génération des DoGs normalisés (valeur en 0 et 1)
        DoGsNormalized = []
        for k in range(len(DoGs)):
            DoGsNormalized.append(ImageManager.normalizeImage(DoGs[k]))

        def _filtrerPointsContraste():
            Log.debug("Demarrage du filtrage par contraste", 1)
            realPoints = []

            for i_sigma in range(1, len(DoGs) - 1):
                # On fait une boucle sur l'ensemble des pixels, bords exclus afin de ne pas avoir a faire du cas par cas
                Log.debug("Traitement du sigma " + str(i_sigma) + " : " + str(sigmas[i_sigma]), 2)
                for row in range(1, height - 1):
                    for col in range(1, width - 1):
                        if abs(DoGsNormalized[i_sigma][row, col]) >= seuil_contraste:
                            realPoints.append((row, col, i_sigma))

            return realPoints

        def _detectionExtremums(candidats):
            Log.debug("Demarrage de la détection des extremums", 1)
            extremums = []

            for c in candidats:
                (row, col, i_sigma) = c

                group_max, group_min = -np.inf, np.inf

                for o in range(-1, 2):
                    elt = DoGs[i_sigma + o][(row - 1):(row + 1) + 1, (col - 1):(col + 1) + 1]
                    group_max, group_min = max(group_max, elt.max()), min(group_min, elt.min())

                # Si le point est un extremum de la region, c'est un point candidat
                if DoGs[i_sigma][row, col] in [group_max, group_min]:
                    extremums.append(c)

            return extremums

        def _filtrerPointsArete(candidats):
            Log.debug("Demarrage du filtrage des arêtes", 1)
            realPoints = []

            Dx, Dy, Dxx, Dyy, Dxy = {}, {}, {}, {}, {}

            Fx = np.matrix('-1 0 1;-2 0 2;-1 0 1')
            Fy = np.matrix('1 2 1;0 0 0;-1 -2 -1')

            for k in range(1, len(DoGs) - 1):
                Dx[k] = Filter.convolve2D(DoGsNormalized[k], Fx)
                Dy[k] = Filter.convolve2D(DoGsNormalized[k], Fy)

            for k in range(1, len(DoGs) - 1):
                Dxx[k] = Filter.convolve2D(Dx[k], Fx)
                Dyy[k] = Filter.convolve2D(Dy[k], Fy)
                Dxy[k] = (Filter.convolve2D(Dx[k], Fy) + Filter.convolve2D(Dy[k], Fx)) / 2

            # On calcul la Hessienne
            for c in candidats:
                (row, col, i_sigma) = c

                Tr = Dxx[i_sigma][row, col] + Dyy[i_sigma][row, col]
                Det = (Dxx[i_sigma][row, col] * Dyy[i_sigma][row, col]) - (Dxy[i_sigma][row, col] ** 2)

                R = (Tr ** 2) / Det

                rapport = ((r_courb_principale + 1) ** 2) / r_courb_principale
                if abs(R) < rapport:
                    realPoints.append(c)

            return realPoints

        def _assignOrientation(candidats):
            Log.debug("Demarrage de l'assignation d'orientation", 1)
            realPoints = []
            # On creer le slice de l'histogramme
            H_slice = np.linspace(0, 2 * np.pi, 36 + 1)  # 37 valeurs, donc 36 intervalles

            for c in candidats:
                (row, col, i_sigma) = c

                # Pour n, on va chercher dans un voisinage de row-n:row+n, col-n:col+n pixels soit (n+1)**2 pixels
                taille_voisinage = int(sigmas[i_sigma] * 3)
                H = np.zeros(36)

                # Selection des points du voisinage en faisant attention aux bords
                rowMax, colMax, rowMin, colMin = min(height - 2, row + taille_voisinage), \
                                                 min(width - 2, col + taille_voisinage), \
                                                 max(1, row - taille_voisinage), \
                                                 max(1, col - taille_voisinage)

                h, b, g, d = octaves[i_sigma][(rowMin - 1):(rowMax - 1) + 1, colMin:colMax + 1], \
                             octaves[i_sigma][(rowMin + 1):(rowMax + 1) + 1, colMin:colMax + 1], \
                             octaves[i_sigma][rowMin:rowMax + 1, (colMin - 1):(colMax - 1) + 1], \
                             octaves[i_sigma][rowMin:rowMax + 1, (colMin + 1):(colMax + 1) + 1]

                g1, g2 = d - g, b - h

                # Calcul des amplitudes des gradients et de l'orientation
                M = np.sqrt(np.power(g1, 2) + np.power(g2, 2))
                A = np.arctan2(g1, g2)  # On utilise atan2 comme spécifié dans l'article en anglais
                A = (A + 2 * np.pi) % (2 * np.pi)  # Opération permettant de revenir dans l'interval [0:2pi]

                # On applique une fenetre gaussienne afin de diminuer les l'impact des points éloignés du point clé
                gaussian = Filter.createGaussianFilter(taille_voisinage, 1.5 * sigmas[i_sigma])
                gaussian = gaussian[taille_voisinage - (row - rowMin):taille_voisinage + (rowMax - row) + 1,
                           taille_voisinage - (col - colMin):taille_voisinage + (colMax - col) + 1]

                M = M * gaussian  # Produit terme à terme

                # Analyse des résultats, on aplatit le carré de matrice pour pouvoir lister les angles
                Ms, As = M.flat, A.flat

                for k, angle in enumerate(As):
                    for si in range(36):
                        if H_slice[si] < angle <= H_slice[si + 1]:
                            H[si] += Ms[k]
                            break

                # On selectionne les angles ayant plus de 80% de la valeur maximale
                mH, angles = np.max(H), []
                for k, a in enumerate(H):
                    if H[k] / mH >= 0.80:
                        angles.append(H_slice[k + 1])

                for a in angles:
                    realPoints.append((row, col, i_sigma, a))

            return realPoints

        # On initialise la tableau d'analyse pour l'analyseur
        pyramid_analyzer = kwargs.get("pyramid_analyzer", None)
        analyseur = OctaveAnalyzer(octave_nb, width, height) if pyramid_analyzer else None
        elements = {} if analyseur else None

        # Filtrage des points de faible contrastes
        candidats = _filtrerPointsContraste()
        Log.debug(str(len(candidats)) + " points", 1)
        if analyseur:
            elements["kp_after_contrast_limitation"] = copy.deepcopy(candidats)

        # Detection des extremums
        candidats = _detectionExtremums(candidats)
        if analyseur:
            elements["kp_after_extremum_detection"] = copy.deepcopy(candidats)

        # Filtrage des points sur les arêtes
        Log.debug(str(len(candidats)) + " points", 1)
        candidats = _filtrerPointsArete(candidats)
        if analyseur:
            elements["kp_after_hessian_filter"] = copy.deepcopy(candidats)

        # Assignation de l'orientation des points
        Log.debug(str(len(candidats)) + " points", 1)
        candidats = _assignOrientation(candidats)
        if analyseur:
            elements["kp_after_orientation_assignation"] = copy.deepcopy(candidats)

        # Packaging des points clés et des outils d'analyse
        if analyseur:
            analyseur.finalKeypoints = copy.deepcopy(candidats)
            analyseur.elements = elements
            pyramid_analyzer.addOctaveAnalyzer(analyseur)

        Log.debug(str(len(candidats)) + " points", 1)

        return candidats

    @staticmethod
    def descriptionPointsCles(image_initiale, points_cles):
        height, width = ImageManager.getDimensions(image_initiale)
        descripteurs = []

        H_angle = np.linspace(0, 2 * np.pi, 8 + 1)
        nb_point_cles = len(points_cles)

        LIMIT = int(8 * np.sqrt(2) + 8)

        for n, (row, col, sigma, a) in enumerate(points_cles):
            Utils.updateProgress(float(n) / nb_point_cles)

            # On élimine les points clés trop pret du bord
            if not (row < 9 or col < 9 or height - row < 9 or width - col < 9):
                rowMax, colMax, rowMin, colMin = min(height - 1, row + LIMIT), \
                                                 min(width - 1, col + LIMIT), \
                                                 max(0, row - LIMIT), \
                                                 max(0, col - LIMIT)

                image_travail = image_initiale[rowMin:(rowMax + 1), colMin:(colMax + 1)]
                row, col = row - rowMin, col - colMin

                # On travail sur l'image lissée avec le
                # paramètre de facteur d'échelle le plus proche
                # de celui du point-clé considéré
                image_travail = ImageManager.applyGaussianFilter(image_travail, sigma)
                image_travail = ImageManager.rotate(image_travail, -a * 180 / np.pi, (row, col))

                # On dessine un carré de coté 16x16
                zone_etude_g = image_travail[(row - 8):(row + 8), (col - 8) - 1:(col + 8) - 1]
                zone_etude_d = image_travail[(row - 8):(row + 8), (col - 8) + 1:(col + 8) + 1]
                zone_etude_h = image_travail[(row - 8) - 1:(row + 8) - 1, (col - 8):(col + 8)]
                zone_etude_b = image_travail[(row - 8) + 1:(row + 8) + 1, (col - 8):(col + 8)]

                zone_etude_g1, zone_etude_g2 = zone_etude_d - zone_etude_g, zone_etude_b - zone_etude_h

                # Calcul des amplitude des gradients et de l'orientation
                M = np.sqrt(np.power(zone_etude_g1, 2) + np.power(zone_etude_g2, 2))
                A = np.arctan2(zone_etude_g1, zone_etude_g2)
                A = (A + 2 * np.pi) % (2 * np.pi)  # Opération permettant de revenir dans l'interval [0:2pi]

                # On applique une fenêtre gaussienne centrée sur le point clé
                gaussian = Filter.createGaussianFilter(8, 1.5 * sigma)
                gaussian = gaussian[:16, :16]

                M = M * gaussian  # Produit terme à terme

                # On fait l'étude sur chaque carrés de coté 4x4 contenus dans la zone de travail
                H_container = []
                for i in range(4):
                    for j in range(4):
                        # On étudie un carré de coté 4x4
                        As = A[i * 4:i * 4 + 4, j * 4:j * 4 + 4]
                        Ms = M[i * 4:i * 4 + 4, j * 4:j * 4 + 4]

                        Mf, Af = Ms.flat, As.flat

                        H = [0.0] * 8

                        for k, angle in enumerate(As.flat):
                            for si in range(8):
                                if angle <= H_angle[si + 1]:
                                    H[si] += float(Mf[k])
                                    break

                        H_container.append(H)

                # On concatène les histogrammes
                H_final = H_container[0]
                for elt in H_container[1::]:
                    H_final = H_final + elt

                H_final = np.array(H_final)
                # On normalise
                H_final = H_final / float(np.max(H_final))

                # On plafonne a 0.2
                for k in range(len(H_final)):
                    if H_final[k] > 0.2:
                        H_final[k] = 0.2

                #  On renormalise
                H_final = H_final / float(np.max(H_final))

                descripteur = np.concatenate((np.array([row, col]), H_final))
                descripteurs.append(descripteur)

        print("")  # Debuff pour la barre de chargement

        return descripteurs
