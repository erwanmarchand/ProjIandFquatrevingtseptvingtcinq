# -*- coding: utf-8 -*-
import copy


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
                rPoints.append((x * (2 ** octave_nb), y * (2 ** octave_nb), j, 1))

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
