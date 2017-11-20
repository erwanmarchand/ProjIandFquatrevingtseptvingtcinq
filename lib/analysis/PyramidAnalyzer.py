# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.analysis.OctaveAnalyzer import *
from lib.Utils import *
import lib.analysis.Utils as UtilsAnalysis

import os


class PyramidAnalyzer:
    def __init__(self, outpath="out/"):
        self.outpath = outpath
        self.orginalPicture = None
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
        else:
            # Si il existe, on le vide
            for root, dirs, files in os.walk(outpath, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(outpath)
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
            UtilsAnalysis.Utils.showPyramid(self.imagePyramid, self.sigmas)
            plt.savefig(self.outpath + "images_pyramid.png")
            UtilsAnalysis.Utils.showPyramid(self.dogPyramid, self.sigmas)
            plt.savefig(self.outpath + "dogs_pyramid.png")

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

                    plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))
                    img = PyramidAnalyzer.showKeyPoints(self.orginalPicture, candidates)
                    plt.imshow(img, cmap="gray")
                    plt.title("Octave " + str(i))

                plt.savefig(self.outpath + p)

        def saveOctaves():
            for i, oa in enumerate(self.octavesAnalyzers):
                plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))
                img = PyramidAnalyzer.showKeyPoints(self.orginalPicture, oa.finalKeypoints)
                plt.imshow(img, cmap="gray")
                plt.title("Octave " + str(i))

            # Affichage des points par octaves
            plt.savefig(self.outpath + "final_octaves.png")

        # Lancement de l'analyse
        savePyramids()
        saveCandidats()
        saveOctaves()

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image
