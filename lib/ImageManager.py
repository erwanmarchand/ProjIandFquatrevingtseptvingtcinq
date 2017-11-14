# -*- coding: utf-8 -*-

import cv2
from Filter import *


class ImageManager:
    @staticmethod
    def loadMatrix(path, **kwargs):
        flag = kwargs.get('flag', cv2.IMREAD_GRAYSCALE)
        data = cv2.imread(path, flag)

        return data

    @staticmethod
    def getGreyscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def getOctave(image_matrix, octave):
        p = float(2 ** octave)

        return cv2.resize(image_matrix, (0, 0), fx=(1.0 / p), fy=(1.0 / p))

    @staticmethod
    def applyGaussianFilter(image, sigma):
        filter = Filter.createGaussianFilter(3, sigma)
        return Filter.applyFilter(image, filter)

    @staticmethod
    def printKeyPoint(image, keypoint):
        pass


    @staticmethod
    def makeDifference(img1, img2):
        return img1 - img2


if __name__ == '__main__':
    pass
