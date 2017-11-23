# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

from lib.Filter import *

GAUSSIAN_PARAMETER = 2


class ImageManager:
    @staticmethod
    def loadMatrix(path, *args):
        data = cv2.imread(path, *args)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        return data

    @staticmethod
    def writeImage(image, path):
        cv2.imwrite(path, image)

    @staticmethod
    def normalizeImage(image):
        if np.max(image) > 1:  # On vérifie que l'image n'est pas déja normalisée
            return image / max(np.max(image), np.min(image))
        else:
            return image

    @staticmethod
    def getGreyscale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def getOctave(image_matrix, octave):
        p = float(2 ** octave)

        return cv2.resize(image_matrix, (0, 0), fx=(1.0 / p), fy=(1.0 / p))

    @staticmethod
    def applyGaussianFilter(image, sigma):
        filter = Filter.createGaussianFilter(int(sigma * 3), sigma)
        return Filter.applyFilter(image, filter)

    @staticmethod
    def showKeyPoint(image, keypoint, **kwargs):
        try:
            h, w, _ = image.shape
        except ValueError:
            h, w = image.shape

        radius = (keypoint[2] ** 1.5) * 8 * min(w, h) / 1024

        COLOR = [(keypoint[2] * 1000) % 256, (keypoint[2] * 333) % 256, (keypoint[2] * 666) % 256]
        image = cv2.circle(image, (keypoint[1], keypoint[0]), int(radius), COLOR, 4)

        # On dessine la fleche représentant l'orientation du point clé
        try:
            kp3 = keypoint[3]
        except IndexError:
            kp3 = None

        if kp3 is not None:
            image = cv2.line(image,
                             (keypoint[1], keypoint[0]),
                             (keypoint[1] + int(np.cos(keypoint[3]) * radius),
                              keypoint[0] - int(np.sin(keypoint[3]) * radius)),
                             COLOR, 2)

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
