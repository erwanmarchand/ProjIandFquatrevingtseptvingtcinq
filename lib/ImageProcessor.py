# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.debug.Log import *


class ImageProcessor:
    @staticmethod
    def findKeypoints(image, s, nb_octaves, **kwargs):
        # On construit la pyramide des gaussiennes
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image, s, nb_octaves, **kwargs)

        realPoints = []

        for i in range(len(octaves)):
            points = ExtremaDetector.detectionPointsCles(
                    DoGs[i],
                    octaves[i],
                    sigmas,
                    0.06,
                    0.6,
                    1 / (2 ** i)
            )

            # On fait un rescale des points clés
            for kp in points:
                (x, y, s, a) = kp
                realPoints.append((x * (2 ** i), y * (2 ** i), s, a))

            if kwargs.get("verbose", False):
                Log.debug("Nombre de points pour l'octave " + str(i) + " : " + str(len(points)))

        return realPoints

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image
