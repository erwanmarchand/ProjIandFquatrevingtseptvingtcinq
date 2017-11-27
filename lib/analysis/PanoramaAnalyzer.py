# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.analysis.OctaveAnalyzer import *
from lib.Utils import *
from lib.debug.Log import *

import lib.analysis.Utils as UtilsAnalysis

import os
import copy
import matplotlib
import matplotlib.pyplot as plt

DPI = 1500
EXTENSION = "png"


class PanoramaAnalyzer:
    def __init__(self, outpath="out/"):
        self.outpath = outpath if outpath[-1] == "/" else outpath + "/"

        self.originalLeftPicture = None
        self.keyPointsLeftPicture = []

        self.originalRightPicture = None
        self.keyPointsRightPicture = []

        self.friendlyCouples = []

        matplotlib.rcParams.update({'font.size': 5})

        if not os.path.exists(outpath):
            # On creer le dossier de sortie si il n'existe pas
            os.makedirs(outpath)

    def analyze(self):
        functions = [
            self.saveFriendlyCouples
        ]

        # Lancement de l'analyse
        i = 1
        for f in functions:
            i = f(i)
            plt.clf()
            plt.cla()

    def saveToFile(self, name):
        plt.savefig(self.outpath + name + "." + EXTENSION, bbox_inches='tight', format=EXTENSION, dpi=DPI)

    def saveFriendlyCouples(self, fi):
        Log.debug("Génération des images avec couples de points clés amis")
        plt.figure(fi + 1)

        candidatesLeft = []
        candidatesRight = []
        for i in range (0, len(self.friendlyCouples)):
            candidatesLeft.append(self.friendlyCouples[i][0])
            candidatesRight.append(self.friendlyCouples[i][1])

        img = PanoramaAnalyzer.showKeyPoints(copy.deepcopy(self.originalLeftPicture), candidatesLeft)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.friendlyCouples)))

        # Affichage des points par octaves
        self.saveToFile("friendly_keypoints_left_image")
        plt.clf()
        plt.cla()


        plt.figure(fi + 2)

        img = PanoramaAnalyzer.showKeyPoints(copy.deepcopy(self.originalRightPicture), candidatesRight)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.friendlyCouples)))

        # Affichage des points par octaves
        self.saveToFile("friendly_keypoints_right_image")
        plt.clf()
        plt.cla()

        Log.debug("Fin génération images")

        return fi + 2

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image
   