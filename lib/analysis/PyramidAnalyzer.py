# -*- coding: utf-8 -*-
from lib.ImageManager import *
from lib.analysis.OctaveAnalyzer import *


class PyramidAnalyzer:
    def __init__(self, originalPicture):
        self.orginalPicture = originalPicture

        self.sigmas = []
        self.imagePyramid = None
        self.dogPyramid = None
        self.octavesAnalyzers = []
        self.keypoints = []

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
        for i, oa in enumerate(self.octavesAnalyzers):
            plt.subplot("1" + str(len(self.octavesAnalyzers)) + str(i))
            img = PyramidAnalyzer.showKeyPoints(self.orginalPicture, oa.finalKeypoints)
            plt.imshow(img, cmap="gray")
            plt.title("Octave " + str(i))

        # Affichage des points par octaves
        plt.show()

    @staticmethod
    def showKeyPoints(image, keypoints):
        for k, keypoint in enumerate(keypoints):
            image = ImageManager.showKeyPoint(image, keypoint)

        return image