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
        self.keyPointsLeftPicture = None

        self.originalRightPicture = None
        self.keyPointsRightPicture = None

        self.friendlyCouples = None

        matplotlib.rcParams.update({'font.size': 5})

        if not os.path.exists(outpath):
            # On creer le dossier de sortie si il n'existe pas
            os.makedirs(outpath)

    def analyze(self):
        functions = [
            self.saveTest
        ]

        # Lancement de l'analyse
        i = 1
        for f in functions:
            i = f(i)
            plt.clf()
            plt.cla()

    def saveToFile(self, name):
        plt.savefig(self.outpath + name + "." + EXTENSION, bbox_inches='tight', format=EXTENSION, dpi=DPI)

    def saveTest(self, fi):
        Log.debug("Génération de l'image test")
        plt.figure(fi + 1)

        candidates = self.keyPointsLeftPicture
        img = PanoramaAnalyzer.showKeyPoints(copy.deepcopy(self.originalLeftPicture), candidates)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.keyPointsLeftPicture)))

        # Affichage des points par octaves
        self.saveToFile("test")
        plt.clf()
        plt.cla()

        return fi + 1

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image
   