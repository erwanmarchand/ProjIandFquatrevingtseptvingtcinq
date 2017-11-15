# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.debug.Log import *


class ImageProcessor:
    @staticmethod
    def findKeypoints(image, s, nb_octaves, **kwargs):
        # On construit la pyramide des gaussiennes
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image, s, nb_octaves,
                                                                       verbose=kwargs.get("verbose", False))

        for i in range(len(octaves)):
            points = ExtremaDetector.detectionPointsCles(
                DoGs[i],
                octaves[i],
                sigmas,
                0.03,
                0.6,
                1 / (2 ** i)
            )

            if kwargs.get("verbose", False):
                Log.info("Nombre de points pour l'octave " + str(i) + " : " + str(len(points)))

    def showKeyPoints(self, image, keypoints):
        for keypoint in keypoints:
            image = ImageManager.showKeyPoint(image, keypoint)

        return image