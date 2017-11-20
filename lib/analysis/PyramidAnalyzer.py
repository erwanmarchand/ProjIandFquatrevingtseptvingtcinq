# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.analysis.OctaveAnalyzer import *
from lib.Utils import *
from lib.debug.Log import *

import lib.analysis.Utils as UtilsAnalysis

import os

DPI = 800

class PyramidAnalyzer:
    def __init__(self, outpath="out/"):
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
        def savePyramids():
            Log.debug("Génération de l'image pour la pyramide des gaussiennes")
            UtilsAnalysis.Utils.showPyramid(self.imagePyramid, self.sigmas)
            plt.savefig(self.outpath + "images_pyramid.jpg", format='jpg', dpi=DPI)

            Log.debug("Génération de l'image pour la pyramide des différences de gaussiennes")
            UtilsAnalysis.Utils.showPyramid(self.dogPyramid, self.sigmas)
            plt.savefig(self.outpath + "dogs_pyramid.jpg", format='jpg', dpi=DPI)

        def saveCandidats():
            if len(self.octavesAnalyzers) == 0:
                raise Exception("Aucun analyseur d'octave trouvé")

            phases = self.octavesAnalyzers[0].elements.keys()

            for p in phases:
                Log.debug("Génération de l'image pour : " + p)
                for oa in self.octavesAnalyzers:
                    i = oa.octaveNb

                    candidates = oa.elements[p]
                    candidates = Utils.adaptKeypoints(candidates, i)
                    candidates = Utils.adaptSigmas(candidates, self.sigmas)

                    plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))
                    img = PyramidAnalyzer.showKeyPoints(self.originalPicture, candidates)
                    plt.imshow(img)
                    plt.title("Octave " + str(i))

                plt.savefig(self.outpath + p + ".jpg", format='jpg', dpi=DPI)

        def saveOctaves():
            for oa in self.octavesAnalyzers:
                i = oa.octaveNb
                plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))
                img = PyramidAnalyzer.showKeyPoints(self.originalPicture, oa.finalKeypoints)
                plt.imshow(img)
                plt.title("Octave " + str(i))

            # Affichage des points par octaves
            plt.savefig(self.outpath + "final_octaves.jpg", format='jpg', dpi=DPI)

        # Lancement de l'analyse
        savePyramids()
        saveCandidats()
        saveOctaves()

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image
