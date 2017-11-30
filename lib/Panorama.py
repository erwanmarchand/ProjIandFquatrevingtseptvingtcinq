# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import numpy as np
import copy


class Panorama:
    def __init__(self):
        pass

    @staticmethod
    def getSIFTPoints(imgLeft, imgRight, **kwargs):
        s = 3
        octave = 3

        # Chargement de l'analyseur
        panoramaAnalyzer = kwargs.get("panorama_analyzer", None)

        # On convertit l'image en nuances de gris pour travailler dessus
        Log.debug("Conversion des images en niveaux de gris")
        imgLeftGreyscale = ImageManager.getGreyscale(imgLeft)
        imgRightGreyscale = ImageManager.getGreyscale(imgRight)

        # On vérifie que le nombre d'octave n'est pas trop grand
        octave_debug = min(int(np.log2(imgLeftGreyscale.shape[0])), int(np.log2(imgLeftGreyscale.shape[1])), octave)
        octave_debug = min(int(np.log2(imgRightGreyscale.shape[0])), int(np.log2(imgRightGreyscale.shape[1])),
                           octave_debug)

        if octave_debug != octave:
            Log.info("Le nombre d'octave a été changé à " + str(
                int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
            octave = octave_debug

        # On applique l'algorithme
        keypointsLeft = ImageProcessor.findKeypoints(imgLeftGreyscale, s, octave,
                                                     verbose=DEBUG,
                                                     pyramid_analyzer=None)
        keypointsRight = ImageProcessor.findKeypoints(imgRightGreyscale, s, octave,
                                                      verbose=DEBUG,
                                                      pyramid_analyzer=None)

        if panoramaAnalyzer:
            panoramaAnalyzer.keyPointsLeftPicture = copy.deepcopy(keypointsLeft)
            panoramaAnalyzer.keyPointsRightPicture = copy.deepcopy(keypointsRight)

        return (keypointsLeft, keypointsRight)

    @staticmethod
    def distanceInterPoints(points_image1, points_image2, **kwargs):
        def _distanceEuclidean(point1, point2):
            D = np.power((point1[2::] - point2[2::]), 2)
            result = np.sqrt(np.sum(D))

            return result

        nbr_key_points_img_left, nbr_key_points_img_right = len(points_image1), len(points_image2)

        Log.debug("Calcul de la matrice de taille : " + str(nbr_key_points_img_left) + " x "
                  + str(nbr_key_points_img_right) + " des distances entre points clés")

        euclidean_dist = np.zeros((nbr_key_points_img_left, nbr_key_points_img_right))

        for i in range(0, euclidean_dist.shape[0]):
            if i % int(euclidean_dist.shape[0] / 20) == 0:
                Log.debug(str(round(float(i) / float(euclidean_dist.shape[0]) * 100, 2)) + " %", 1)

            for j in range(0, euclidean_dist.shape[1]):

                euclidean_dist[i][j] = _distanceEuclidean(points_image1[i], points_image2[j])

        return euclidean_dist

    @staticmethod
    def getFriendlyCouples(imgLeft, imgRight, n, **kwargs):
        # Chargement de l'analyseur
        panoramaAnalyzer = kwargs.get("panorama_analyzer", None)

        friendlyPoints = []

        (SIFTPointsLeft, SIFTPointsRight) = Panorama.getSIFTPoints(imgLeft, imgRight,
                                                                   panorama_analyzer=panoramaAnalyzer,
                                                                   verbose=kwargs.get("verbose", False))

        matrixDistances = Panorama.distanceInterPoints(SIFTPointsLeft, SIFTPointsRight,
                                                       panorama_analyzer=panoramaAnalyzer,
                                                       verbose=kwargs.get("verbose", False))

        Log.debug("Recherche des couples amis")

        maxValue = matrixDistances.max()

        for index in range(0, n):
            minValue = matrixDistances.min()
            minPosition = np.where(matrixDistances == minValue)
            iMin, jMin = minPosition[0][0], minPosition[1][0]

            friendlyPoints.append((SIFTPointsLeft[iMin], SIFTPointsRight[jMin]))
            matrixDistances[iMin][jMin] = maxValue

        if panoramaAnalyzer:
            panoramaAnalyzer.friendlyCouples = copy.deepcopy(friendlyPoints)

        return friendlyPoints

    @staticmethod
    def getMatriceA(friendlyPoints):
        numberOfFriendlyPoints = len(friendlyPoints)
        matriceA = []
        for i in range(0, numberOfFriendlyPoints):
            # Descripteurs sous la forme [ y, x, desc SIFT ]
            # Image droite (coordonnées de départ des points)
            xn = friendlyPoints[i][1][1]
            yn = friendlyPoints[i][1][0]
            # Image gauche (coordonnées d'arrivée des points)
            xpn = friendlyPoints[i][0][1]
            ypn = friendlyPoints[i][0][0]
            matriceA.append([xn, yn, 1, 0, 0, 0, -xpn * xn, -xpn * yn, -xpn])
            matriceA.append([0, 0, 0, xn, yn, 1, -ypn * xn, -ypn * yn, -ypn])
        return matriceA

    @staticmethod
    def getTransformMatrix(A):
        AT = np.transpose(A)
        B = np.dot(AT, A)
        (valPropres, vectPropres) = np.linalg.eig(B)
        indexValMin = np.argmin(valPropres)
        Hflatten = vectPropres[:, indexValMin]
        # Hflatten = vectPropres[indexValMin]
        HflattenNorm = Hflatten / Hflatten[len(Hflatten) - 1]
        Hnorm = HflattenNorm.reshape(3, 3)
        return Hnorm
