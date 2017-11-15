# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

from lib.Filter import *

GAUSSIAN_PARAMETER = 2


class ImageManager:
    @staticmethod
    def loadMatrix(path, **kwargs):
        flag = kwargs.get('flag', cv2.IMREAD_GRAYSCALE)
        data = cv2.imread(path, flag)

        return data

    @staticmethod
    def normalizeImage(image):
        if np.max(image) > 1:  # On vérifie que l'image n'est pas déja normalisée
            return image / np.max(image)
        else:
            return image

    @staticmethod
    def getGreyscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def getOctave(image_matrix, octave):
        p = float(2 ** octave)

        return cv2.resize(image_matrix, (0, 0), fx=(1.0 / p), fy=(1.0 / p))

    @staticmethod
    def applyGaussianFilter(image, sigma):
        filter = Filter.createGaussianFilter(GAUSSIAN_PARAMETER, sigma)
        return Filter.applyFilter(image, filter)

    @staticmethod
    def showKeyPoint(image, keypoint):
        BLUE = [255, 0, 0]

        radius = keypoint[2] * 5
        image = cv2.circle(image, (keypoint[0], keypoint[1]), int(radius), color=BLUE)
        image = cv2.arrowedLine(image,
                                (keypoint[0], keypoint[1]),
                                (int(np.sin(keypoint[3]) * radius), int(np.cos(keypoint[3]) * radius)),
                                BLUE)

        return image

    @staticmethod
    def makeDifference(img1, img2):
        return img1 - img2

    @staticmethod
    def showImage(image, **kwargs):
        plt.imshow(image, **kwargs)
        plt.show()

    @staticmethod
    def getDimensions(image):
        return image.shape


if __name__ == '__main__':
    pass
