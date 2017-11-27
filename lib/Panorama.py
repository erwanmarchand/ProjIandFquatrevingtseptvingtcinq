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

            result = 0
            for k in range(2, len(point1)):
                result = result + ((point1[k] - point2[k]) ** 2)
            result = np.sqrt(result)
            return result

        nbrKeyPointsImgLeft = len(points_image1)
        nbrKeyPointsImgRight = len(points_image2)

        Log.debug("Calcul de la matrice de taille : "+str(nbrKeyPointsImgLeft)+" x "+str(nbrKeyPointsImgRight)+" des distances entre points clés")

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

        Log.debug("Recherche des couples amis")

        matrixDistances = copy.deepcopy(distEuc)

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

    @staticmethod
    def getMatriceA(friendlyPoints) :
        numberOfFriendlyPoints = len(friendlyPoints)
        matriceA = []
        for i in range (0,numberOfFriendlyPoints) :
            matriceA.append([friendlyPoints[i][0][0],friendlyPoints[i][0][1],1,0,0,0,-friendlyPoints[i][1][0]*friendlyPoints[i][0][0],-friendlyPoints[i][1][0]*friendlyPoints[i][0][1],-friendlyPoints[i][1][0]])
            matriceA.append([0,0,0,friendlyPoints[i][0][0],friendlyPoints[i][0][1],1,-friendlyPoints[i][1][1]*friendlyPoints[i][0][0],-friendlyPoints[i][1][1]*friendlyPoints[i][0][1],-friendlyPoints[i][1][1]])
        return matriceA

    @staticmethod
    def getTransformMatrix(A) :
        AT = np.transpose(A)
        B = np.dot(AT,A)
        (valPropres,vectPropres) = np.linalg.eig(B)
        Hflatten = vectPropres[:,np.argmin(valPropres)]
        Hflatten = Hflatten/Hflatten[len(Hflatten)-1]
        Hnorm = Hflatten.reshape(3,3)
        return Hnorm