# -*- coding: utf-8 -*-
import numpy as np
import math
import scipy.signal as signal


class Filter:
    @staticmethod
    def applyFilter(image, filter):
        return signal.convolve2d(image, filter)

    @staticmethod
    def createGaussianFilter(P, sigma):
        matrix = np.zeros((2 * P + 1, 2 * P + 1))
        som = 0
        for m in range(-P, P + 1):
            for n in range(-P, P + 1):
                matrix[m + P][n + P] = math.exp(-(n ** 2 + m ** 2) / (2 * (sigma ** 2)))
                som += matrix[m + P][n + P]
        matrix = matrix / som

        return matrix
