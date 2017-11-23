from lib.ImageProcessor import *
from lib.analysis.PyramidAnalyzer import *
from lib.debug.Log import *

import lib.debug.Log as Log_file
import numpy as np

class Panorama:
    def __init__(self):
        pass
        
    @staticmethod
    def getSIFTPoints(imgLeft, imgRight):
        s = 3
        octave = 3

        # On convertit l'image en nuances de gris pour travailler dessus
        Log.debug("Conversion des images en niveaux de gris")
        imgLeftGreyscale = ImageManager.getGreyscale(imgLeft)
        imgRightGreyscale = ImageManager.getGreyscale(imgRight)

        # On vérifie que le nombre d'octave n'st pas trop grand
        octave_debug = min(int(np.log2(imgLeftGreyscale.shape[0])), int(np.log2(imgLeftGreyscale.shape[1])), octave)
        if octave_debug != octave:
            Log.info("Le nombre d'octave a été changé à " + str(
                int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
            octave = octave_debug

        octave_debug = min(int(np.log2(imgRightGreyscale.shape[0])), int(np.log2(imgRightGreyscale.shape[1])), octave)
        if octave_debug != octave:
            Log.info("Le nombre d'octave a été changé à " + str(
                int(octave_debug)) + " afin d'éviter les problèmes de redimensionnement")
            octave = octave_debug

        # On applique l'algorithme

        # TODO : decomment next lines and remove DEBUG Ones
        #keypointsLeft = ImageProcessor.findKeypoints(imgLeftGreyscale, s, octave,
        #                                verbose=DEBUG,
        #                                 pyramid_analyzer=None)
        #keypointsRight= ImageProcessor.findKeypoints(imgRightGreyscale, s, octave,
        #                                 verbose=DEBUG,
        #                                 pyramid_analyzer=None)

        #DEBUG
        keypointsLeft = np.trunc(1000 * np.random.rand(10,130))
        keypointsRight = np.trunc(1000 *  np.random.rand(15,130))
        for i in range(0, 4):
            keypointsLeft[i] = keypointsRight[i]
            keypointsLeft[i][1] = keypointsRight[i][1] + 1000 

        return (keypointsLeft,keypointsRight)

    @staticmethod
    def distanceInterPoints(points_image1, points_image2):

        def _distanceEuclidean(point1, point2):

            result = 0
            for i in range(2, point1.size):
                result = result + ((point1[i]-point2[i])**2)
            result = np.sqrt(result)
            return result
        
        nbrKeyPointsImgLeft = points_image1.shape[0]
        nbrKeyPointsImgRight = points_image2.shape[0]

        euclideanDist = np.zeros((nbrKeyPointsImgLeft,nbrKeyPointsImgRight))

        for i in range(0, euclideanDist.shape[0]):
            for j in range(0, euclideanDist.shape[1]):
                euclideanDist[i][j] = _distanceEuclidean(points_image1[i], points_image2[j])


        return euclideanDist

    