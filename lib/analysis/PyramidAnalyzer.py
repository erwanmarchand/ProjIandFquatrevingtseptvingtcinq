# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.analysis.OctaveAnalyzer import *
from lib.Utils import *
from lib.debug.Log import *

import lib.analysis.Utils as UtilsAnalysis

import os
import copy

DPI = 800
EXTENSION = "png"

class PyramidAnalyzer:
    def __init__(self, outpath="out/"):
        #plt.tight_layout()

        self.outpath = outpath
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

    def analyze(self):
        functions = [
            self.saveCandidats,
            self.savePyramids,
            self.saveOctaves,
            self.generateGraph,
            self.saveFinalKeypoints
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
        plt.savefig(self.outpath + "images_pyramid." + EXTENSION, format=EXTENSION, dpi=DPI)
        plt.clf()
        plt.cla()

        plt.figure(fi + 2)
        Log.debug("Génération de l'image pour la pyramide des différences de gaussiennes")
        UtilsAnalysis.Utils.showPyramid(self.dogPyramid, self.sigmas)
        plt.savefig(self.outpath + "dogs_pyramid." + EXTENSION, format=EXTENSION, dpi=DPI)
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

                candidates = oa.elements[ph]
                candidates = Utils.adaptKeypoints(candidates, i)
                candidates = Utils.adaptSigmas(candidates, self.sigmas)

                plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))
                img = PyramidAnalyzer.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
                plt.imshow(img)
                plt.title("Octave " + str(i + 1) + " - " + str(len(candidates)))

            plt.savefig(self.outpath + ph + "." + EXTENSION, format=EXTENSION, dpi=DPI)
            plt.clf()
            plt.cla()

        return fi

    def saveOctaves(self, fi):
        Log.debug("Génération de l'image finale pour chaque octave")
        plt.figure(fi + 1)
        for oa in self.octavesAnalyzers:
            i = oa.octaveNb
            plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))

            candidates = oa.finalKeypoints
            candidates = Utils.adaptKeypoints(candidates, i)
            candidates = Utils.adaptSigmas(candidates, self.sigmas)

            img = PyramidAnalyzer.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
            plt.imshow(img)
            plt.title("Octave " + str(i + 1) + " - " + str(len(oa.finalKeypoints)))

        # Affichage des points par octaves
        plt.savefig(self.outpath + "final_octaves." + EXTENSION, format=EXTENSION, dpi=DPI)
        plt.clf()
        plt.cla()

        return fi + 1

    def saveFinalKeypoints(self, fi):
        Log.debug("Génération de l'image finale des keypoints")
        plt.figure(fi + 1)

        candidates = self.keypoints
        img = PyramidAnalyzer.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.keypoints)))

        # Affichage des points par octaves
        plt.savefig(self.outpath + "final." + EXTENSION, format=EXTENSION, dpi=DPI)
        plt.clf()
        plt.cla()

        return fi + 1

    def generateGraph(self, i):
        return i + 1
