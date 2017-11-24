# -*- coding: utf-8 -*-
from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import numpy as np


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

        # On vérifie que le nombre d'octave n'st pas trop grand
        octave_debug = min(int(np.log2(imgLeftGreyscale.shape[0])), int(np.log2(imgLeftGreyscale.shape[1])), octave)
        octave_debug = min(int(np.log2(imgRightGreyscale.shape[0])), int(np.log2(imgRightGreyscale.shape[1])), octave_debug)

        if octave_debug != octave:
            Log.info("Le nombre d'octave a été changé à " + str(
                int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
            octave = octave_debug

        # On applique l'algorithme

        # TODO : decomment next lines and remove DEBUG Ones
        keypointsLeft = ImageProcessor.findKeypoints(imgLeftGreyscale, s, octave,
                                                     verbose=DEBUG,
                                                     pyramid_analyzer=None)
        keypointsRight = ImageProcessor.findKeypoints(imgRightGreyscale, s, octave,
                                                      verbose=DEBUG,
                                                      pyramid_analyzer=None)

        # DEBUG
        # keypointsLeft = np.trunc(1000 * np.random.rand(10,130))
        # keypointsRight = np.trunc(1000 *  np.random.rand(15,130))
        # for i in range(0, 4):
        #    keypointsLeft[i] = keypointsRight[i]
        #    keypointsLeft[i][3] = 4
        #    keypointsLeft[i][1] = keypointsRight[i][1] + 1000 
        # keypointsLeft[0] = keypointsRight[5]
        # keypointsLeft[0][1] = keypointsRight[5][1] + 1000
        # keypointsLeft = keypointsLeft.astype(int)
        # keypointsRight = keypointsRight.astype(int)
        # for i in range(0, 10):
        #    keypointsLeft[i][2] = 3
        # for i in range(0, 15):
        #    keypointsRight[i][2] = 3

        if panoramaAnalyzer:
            panoramaAnalyzer.keyPointsLeftPicture = keypointsLeft.copy()
            panoramaAnalyzer.keyPointsRightPicture = keypointsRight.copy()

        return (keypointsLeft, keypointsRight)

    @staticmethod
    def distanceInterPoints(points_image1, points_image2, **kwargs):

        def _distanceEuclidean(point1, point2):

            result = 0
            for i in range(2, len(point1)):
                result = result + ((point1[i] - point2[i]) ** 2)
            result = np.sqrt(result)
            return result

        nbrKeyPointsImgLeft = len(points_image1)
        nbrKeyPointsImgRight = len(points_image2)

        euclideanDist = np.zeros((nbrKeyPointsImgLeft, nbrKeyPointsImgRight))

        for i in range(0, euclideanDist.shape[0]):
            for j in range(0, euclideanDist.shape[1]):
                euclideanDist[i][j] = _distanceEuclidean(points_image1[i], points_image2[j])

        return euclideanDist

    @staticmethod
    def getFriendlyCouples(imgLeft, imgRight, n, **kwargs):

        # Chargement de l'analyseur
        panoramaAnalyzer = kwargs.get("panorama_analyzer", None)

        friendlyPoints = []

        (SIFTPointsLeft, SIFTPointsRight) = Panorama.getSIFTPoints(imgLeft, imgRight,
                                                                   panorama_analyzer=panoramaAnalyzer,
                                                                   verbose=kwargs.get("verbose", False))

        distEuc = Panorama.distanceInterPoints(SIFTPointsLeft, SIFTPointsRight,
                                               panorama_analyzer=panoramaAnalyzer,
                                               verbose=kwargs.get("verbose", False))

        matrixDistances = distEuc.copy()

        maxValue = matrixDistances.max()

        for index in range(0, n):
            minValue = matrixDistances.min()
            minPosition = np.where(matrixDistances == minValue)
            iMin = minPosition[0][0]
            jMin = minPosition[1][0]
            friendlyPoints.append((SIFTPointsLeft[iMin], SIFTPointsRight[jMin]))
            matrixDistances[iMin][jMin] = maxValue

        panoramaAnalyzer.friendlyCouples = copy.deepcopy(friendlyPoints)

        return friendlyPoints
