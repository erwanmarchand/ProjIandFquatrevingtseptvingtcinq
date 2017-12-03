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
        s, nb_octaves = 3, 3

        # Chargement de l'analyseur
        panoramaAnalyzer = kwargs.get("panorama_analyzer", None)

        pyramidAnalyzerLeft = PyramidAnalyzer("out_panorama_left/") if kwargs.get("analyse_each_image", False) else None
        pyramidAnalyzerRight = PyramidAnalyzer("out_panorama_right/") if kwargs.get("analyse_each_image", False) else None


        # On applique l'algorithme sur chaque images
        keypointsLeft = ImageProcessor.findKeypoints(imgLeft, s, nb_octaves,
                                                        verbose=DEBUG,
                                                        pyramid_analyzer=pyramidAnalyzerLeft)
        keypointsRight = ImageProcessor.findKeypoints(imgRight, s, nb_octaves,
                                                         verbose=DEBUG,
                                                         pyramid_analyzer=pyramidAnalyzerRight)

        if kwargs.get("analyse_each_image", False):
            pyramidAnalyzerLeft.analyze()
            pyramidAnalyzerRight.analyze()

        if panoramaAnalyzer:
            panoramaAnalyzer.keyPointsLeftPicture = copy.deepcopy(keypointsLeft)
            panoramaAnalyzer.keyPointsRightPicture = copy.deepcopy(keypointsRight)

        return keypointsLeft, keypointsRight

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
            Utils.updateProgress(float(i) / float(euclidean_dist.shape[0]))

            for j in range(0, euclidean_dist.shape[1]):
                euclidean_dist[i][j] = _distanceEuclidean(points_image1[i], points_image2[j])

        print("")

        return euclidean_dist

    @staticmethod
    def getFriendlyCouples(imgLeft, imgRight, n, **kwargs):
        # Chargement de l'analyseur
        panoramaAnalyzer = kwargs.get("panorama_analyzer", None)

        friendlyPoints = []

        (SIFTPointsLeft, SIFTPointsRight) = Panorama.getSIFTPoints(imgLeft, imgRight, **kwargs)

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

        ecartKPLeft = np.sqrt(((friendlyPoints[0][0][1] - friendlyPoints[1][0][1]) ** 2) + (
            (friendlyPoints[0][0][0] - friendlyPoints[1][0][0]) ** 2))
        ecartKPRight = np.sqrt(((friendlyPoints[0][1][1] - friendlyPoints[1][1][1]) ** 2) + (
            (friendlyPoints[0][1][0] - friendlyPoints[1][1][0]) ** 2))
        rapport = ecartKPLeft / ecartKPRight
        return (matriceA, rapport)

    @staticmethod
    def getTransformMatrix(A, rapport):
        # AT = np.transpose(A)
        # B = np.dot(AT, A)
        # (valPropres, vectPropres) = np.linalg.eig(B)
        # indexValMin = np.argmin(np.absolute(valPropres))
        # Hflatten = vectPropres[:, indexValMin]
        # HflattenNorm = Hflatten / Hflatten[len(Hflatten) - 1]
        # Hnorm = HflattenNorm.reshape(3, 3)

        U, s, V = np.linalg.svd(A, full_matrices=True)
        Hflatten2 = V[len(V) - 1]
        HflattenNorm2 = Hflatten2 / Hflatten2[len(Hflatten2) - 1]
        Hnorm2 = HflattenNorm2.reshape(3, 3)
        Hnorm2 = Hnorm2 * rapport

        return Hnorm2

    @staticmethod
    def generatePanorama(leftPicture, rightPicture, homographyMatrix, **kwargs):
        # Chargement de l'analyseur
        panoramaAnalyzer = kwargs.get("panorama_analyzer", None)

        xMaxRight = rightPicture.shape[1] - 1
        yMaxRight = rightPicture.shape[0] - 1
        xMaxLeft = leftPicture.shape[1] - 1
        yMaxLeft = leftPicture.shape[0] - 1

        [xMaxRightOnLeft, yMaxRightOnLeft, temp] = np.round(np.dot(homographyMatrix, [xMaxRight, yMaxRight, 1])).astype(
            int)
        xMaxRightOnLeft = int(xMaxRightOnLeft)
        yMaxRightOnLeft = int(yMaxRightOnLeft)

        xMax = max(xMaxLeft, xMaxRightOnLeft)
        yMax = max(yMaxLeft, yMaxRightOnLeft)

        # finalPicture = np.matrix((yMax+1, xMax+1), [0,0,0])
        finalPicture = np.zeros((yMax + 1, xMax + 1, 3))

        for y in range(0, yMaxRight + 1):
            for x in range(0, xMaxRight + 1):
                [xNew, yNew, temp] = np.round((np.dot(homographyMatrix, [x, y, 1]))).astype(int)
                xNew = int(xNew)
                yNew = int(yNew)

                if xNew <= xMax and yNew <= yMax and yNew >= 0 and xNew >= 0:
                    finalPicture[yNew][xNew] = [256, 256, 256] - rightPicture[y][x]

        for y in range(0, yMaxLeft + 1):
            for x in range(0, xMaxLeft + 1):
                finalPicture[y][x] = [256, 256, 256] - leftPicture[y][x]

        if panoramaAnalyzer:
            panoramaAnalyzer.finalPicture = copy.deepcopy(finalPicture)

        return finalPicture
