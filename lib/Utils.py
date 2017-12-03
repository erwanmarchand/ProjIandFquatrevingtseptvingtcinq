# -*- coding: utf-8 -*-
import copy
import time
import sys


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
                rPoints.append((int(x * (2 ** octave_nb)), int(y * (2 ** octave_nb)), j, a))
            except ValueError:
                (x, y, j) = kp
                rPoints.append((int(x * (2 ** octave_nb)), int(y * (2 ** octave_nb)), j, None))

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

    @staticmethod
    def updateProgress(workdone):
        sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100))
        sys.stdout.flush()

if __name__ == '__main__':
    for i in range(0, 101, 10):
        Utils.updateProgress(i / 100.)
        time.sleep(1)
