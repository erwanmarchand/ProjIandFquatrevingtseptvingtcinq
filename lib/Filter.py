# -*- coding: utf-8 -*-
import math

import numpy as np
import scipy.signal as signal


class Filter:
    def __init__(self):
        pass

    @staticmethod
    def applyFilter(image, filter_matrix, **kwargs):
        """
        Applique un filtre à une image
        :param image:           L'image sur laquelle le filtre doit être appliqué
        :param filter_matrix:   La matrice représentant le filtre
        :param kwargs:          Arguments facultatifs de la méthode de convolution
        :return: L'image convoluée
        """

        return signal.convolve2d(image, filter_matrix, kwargs.get("mode", "same"), kwargs.get("boundary", "symm"))

    @staticmethod
    def createGaussianFilter(P, sigma):
        """
        Génère un filtre gaussien
        :param P:       La taille de la matrice à générer
        :param sigma:   Le paramètre du filtre
        :return:        Une matrice de (2P +1)x(2P + 1) représentant un filtre gaussien de paramètre sigma
        """

        matrix = np.zeros((2 * P + 1, 2 * P + 1))
        som = 0
        for m in range(-P, P + 1):
            for n in range(-P, P + 1):
                matrix[m + P][n + P] = math.exp(-(n ** 2 + m ** 2) / (2 * (sigma ** 2)))
                som += matrix[m + P][n + P]
        matrix = matrix / som

        return matrix


if __name__ == '__main__':
    for p in range(6):
        filter = Filter.createGaussianFilter(p, 1.6)
        print("P = " + str(p) + " || " + str(filter.shape[0]) + "x" + str(filter.shape[1]))
        print(filter)
        print("")
