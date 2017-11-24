# -*- coding: utf-8 -*-
import copy
import numpy as np


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def adaptKeypoints(keypoints, octave_nb):
        """
        Adapte les points clé d'une octave en changeant leur echelle pour correspondre à l'image originale
        :param keypoints: 
        :param octave_nb: 
        :param sigmas: 
        :return: 
        """
        rPoints = []
        for kp in keypoints:
            try:
                (x, y, j, a) = kp
                rPoints.append((x * (2 ** octave_nb), y * (2 ** octave_nb), j, a))
            except ValueError:
                (x, y, j) = kp
                rPoints.append((x * (2 ** octave_nb), y * (2 ** octave_nb), j, None))

        return rPoints

    @staticmethod
    def concatenateKeyPoints(keypoints1, keypoints2):
        newList = copy.deepcopy(keypoints1)

        for elt in keypoints2:
            if elt not in newList:
                newList.append(elt)

        return newList

    @staticmethod
    def adaptSigmas(keypoints, sigmas):
        rPoints = []
        for kp in keypoints:
            try:
                (x, y, j, a) = kp
                rPoints.append((x, y, sigmas[j], a))
            except ValueError:
                (x, y, j) = kp
                rPoints.append((x, y, sigmas[j]))

        return rPoints

    # @staticmethod
    # def calculateOrientation(image, (row, col)):
    #     m = np.sqrt((image[row + 1, col] - image[row - 1, col]) ** 2
    #                   + (image[row, col + 1] - image[row, col - 1]) ** 2)
    #     theta = np.arctan2((image[row, col + 1] - image[row, col - 1])
    #                        , (image[row + 1, col] - image[row - 1, col]))
    #
    #     return m, (theta + 2 * np.pi) % (2 * np.pi)
