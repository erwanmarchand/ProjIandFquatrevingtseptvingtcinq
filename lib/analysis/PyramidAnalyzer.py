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


class PyramidAnalyzer:
    def __init__(self, outpath="out/"):
        self.outpath = outpath if outpath[-1] == "/" else outpath + "/"
        self.originalPicture = None
        self.greyscalePicture = None
        self.remasteredPicture = None

        self.sigmas = []
        self.imagePyramid = None
        self.dogPyramid = None
        self.octavesAnalyzers = []
        self.keypoints = []

        if not os.path.exists(outpath):
            # On creer le dossier de sortie si il n'existe pas
            os.makedirs(outpath)

    def setImagePyramid(self, pyramid):
        self.imagePyramid = pyramid

    def setDoGPyramid(self, pyramid):
        self.dogPyramid = pyramid

    def addOctaveAnalyzer(self, octaveAnalyzer):
        if isinstance(octaveAnalyzer, OctaveAnalyzer):
            self.octavesAnalyzers += [octaveAnalyzer]
        else:
            raise Exception("Erreur : Mauvaise classe passée en paramètre (type : " + str(type(octaveAnalyzer)) + ")")

    def saveToFile(self, name):
        matplotlib.rcParams.update({'font.size': 5})

        plt.savefig(self.outpath + name + "." + EXTENSION, bbox_inches='tight', format=EXTENSION, dpi=DPI)

    def analyze(self):
        functions = [
            self.saveCandidats,
            self.savePyramids,
            self.saveOctaves,
            self.generateGraph,
            self.saveFinal
        ]

        # Lancement de l'analyse
        i = 1
        for f in functions:
            i = f(i)
            plt.clf()
            plt.cla()

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image

    def savePyramids(self, fi):
        plt.figure(fi + 1)
        Log.debug("Génération de l'image pour la pyramide des gaussiennes")
        UtilsAnalysis.Utils.showPyramid(self.imagePyramid, self.sigmas)
        self.saveToFile("images_pyramid")
        plt.clf()
        plt.cla()

        plt.figure(fi + 2)
        Log.debug("Génération de l'image pour la pyramide des différences de gaussiennes")
        UtilsAnalysis.Utils.showPyramid(self.dogPyramid, self.sigmas)
        self.saveToFile("dogs_pyramid")
        plt.clf()
        plt.cla()

        return fi + 2

    def saveCandidats(self, fi):
        if len(self.octavesAnalyzers) == 0:
            raise Exception("Aucun analyseur d'octave trouvé")

        for ph in self.octavesAnalyzers[0].elements.keys():
            # Mise a jour du compteur de figure
            fi = fi + 1

            # Démarrage de l'analyse
            Log.debug("Génération de l'image pour : " + ph)
            plt.figure(fi)
            for oa in self.octavesAnalyzers:
                i = oa.octaveNb

                plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i + 1))

                candidates = oa.elements[ph]
                if ph != "kp_after_contrast_limitation":
                    candidates = Utils.adaptKeypoints(candidates, i)
                    img = PyramidAnalyzer.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
                else:
                    height, width, _ = self.originalPicture.shape
                    img = np.zeros((oa.octaveHeight, oa.octaveWidth))
                    for elt in candidates:
                        try:
                            x, y, j = elt
                        except ValueError:
                            x, y, j, a = elt

                        img[x, y] = 1

                candidates = Utils.adaptSigmas(candidates, self.sigmas)

                plt.imshow(img)
                plt.title("Octave " + str(i + 1) + " - " + str(len(candidates)))

                self.saveToFile(ph)
            plt.clf()
            plt.cla()

        return fi

    def saveOctaves(self, fi):
        Log.debug("Génération de l'image finale pour chaque octave")
        plt.figure(fi + 1)
        for oa in self.octavesAnalyzers:
            i = oa.octaveNb
            plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i + 1))

            candidates = oa.finalKeypoints
            candidates = Utils.adaptKeypoints(candidates, i)
            candidates = Utils.adaptSigmas(candidates, self.sigmas)

            img = PyramidAnalyzer.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
            plt.imshow(img)
            plt.title("Octave " + str(i + 1) + " - " + str(len(oa.finalKeypoints)))

        # Affichage des points par octaves
        self.saveToFile("final_octaves")
        plt.clf()
        plt.cla()

        return fi + 1

    def saveFinal(self, fi):
        Log.debug("Génération de l'image finale des keypoints")
        plt.figure(fi + 1)

        candidates = self.keypoints
        img = PyramidAnalyzer.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.keypoints)))

        # Affichage des points par octaves
        self.saveToFile("final")
        plt.clf()
        plt.cla()

        return fi + 1

    def generateGraph(self, i):
        return i + 1
