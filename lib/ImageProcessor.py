# -*- coding: utf-8 -*-

from lib.ExtremaDetector import *
from lib.debug.Log import *

import matplotlib.pyplot as plt


class ImageProcessor:
    @staticmethod
    def findKeypoints(image, s, nb_octaves, **kwargs):
        # On construit la pyramide des gaussiennes
        DoGs, octaves, sigmas = ExtremaDetector.differenceDeGaussienne(image, s, nb_octaves, **kwargs)

        realPoints = []

        for i in range(len(octaves)):
            rPoints = []
            points = ExtremaDetector.detectionPointsCles(
                    DoGs[i],
                    octaves[i],
                    sigmas,
                    0.20,
                    0.6,
                    1 / (2 ** i)
            )

            # On fait un rescale des points clés
            for kp in points:
                try:
                    (x, y, j, a) = kp
                    rPoints.append((x * (2 ** i), y * (2 ** i), sigmas[j], a))
                except ValueError:
                    (x, y, j) = kp
                    rPoints.append((x * (2 ** i), y * (2 ** i), sigmas[j], 1))

            print(len(rPoints))

            if kwargs.get("verbose", False):
                Log.debug("Nombre de points pour l'octave " + str(i) + " : " + str(len(rPoints)))

            plt.subplot("1"+str(len(octaves))+str(i))
            img = ImageProcessor.showKeyPoints(image, rPoints)
            plt.imshow(img, cmap="gray")
            plt.title("Octave " + str(i))

            realPoints.append(np.array(rPoints))

        # Affichage des points par octaves
        plt.show()

        # Concatenation des tableaux
        Log.debug("Concaténation des tableaux de points clés")
        keyPoints = np.array(realPoints).flat
        keyPoints = np.unique(keyPoints)

        img = ImageProcessor.showKeyPoints(image, keyPoints)
        plt.imshow(img, cmap="gray")
        plt.title("Points clés")
        plt.show()


        exit(0)
        return realPoints

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image
