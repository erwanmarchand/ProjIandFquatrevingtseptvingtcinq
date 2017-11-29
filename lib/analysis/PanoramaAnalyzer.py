# -*- coding: utf-8 -*-
from lib.analysis.Analyzer import *
from lib.debug.Log import *

import os
import copy
import matplotlib
import matplotlib.pyplot as plt


class PanoramaAnalyzer(Analyzer):
    def __init__(self, outpath="out/"):
        Analyzer.__init__(self, outpath)

        self.originalLeftPicture = None
        self.keyPointsLeftPicture = []

        self.originalRightPicture = None
        self.keyPointsRightPicture = []

        self.friendlyCouples = []

        matplotlib.rcParams.update({'font.size': 5})

        if not os.path.exists(outpath):
            # On creer le dossier de sortie si il n'existe pas
            os.makedirs(outpath)

    def getFunctions(self):
        return [
            self.saveFriendlyCouples
        ]

    def saveFriendlyCouples(self, fi):
        Log.debug("Génération des images avec couples de points clés amis")
        plt.figure(fi + 1)

        candidatesLeft = []
        candidatesRight = []

        for i in range(0, len(self.friendlyCouples)):
            candidatesLeft.append(self.friendlyCouples[i][0])
            candidatesRight.append(self.friendlyCouples[i][1])

        img = self.showKeyPoints(copy.deepcopy(self.originalLeftPicture), candidatesLeft)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.friendlyCouples)))

        # Affichage des points par octaves
        self.saveToFile("friendly_keypoints_left_image")
        plt.clf()
        plt.cla()

        plt.figure(fi + 2)

        img = self.showKeyPoints(copy.deepcopy(self.originalRightPicture), candidatesRight)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.friendlyCouples)))

        # Affichage des points par octaves
        self.saveToFile("friendly_keypoints_right_image")
        plt.clf()
        plt.cla()

        Log.debug("Fin génération images")

        return fi + 2
