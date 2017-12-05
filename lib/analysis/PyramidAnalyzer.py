# -*- coding: utf-8 -*-
from lib.analysis.OctaveAnalyzer import *
from lib.analysis.Analyzer import *
from lib.Utils import *
from lib.debug.Log import *

import lib.analysis.Utils as UtilsAnalysis

import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import cv2


class PyramidAnalyzer(Analyzer):
    def __init__(self, outpath="out/"):
        Analyzer.__init__(self, outpath)

        self.originalPicture = None
        self.doubledPicture = None
        self.greyscaleDoubledPicture = None

        self.sigmas = []
        self.imagePyramid = None
        self.dogPyramid = None
        self.octavesAnalyzers = []
        self.key_points = []

        self.descriptors = []

        matplotlib.rcParams.update({'font.size': 5})

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

    def getFunctions(self):
        return [
            self.generateAnalysisGraph,
            self.saveCandidats,
            self.savePyramids,
            self.saveOctaves,
            self.saveFinal,
            self.saveCv2Sift,
            self.saveMatrixKeypoints,
            self.saveMatrixDescriptors,
            self.generateCSVArrayAnalysis
        ]

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

                plt.subplot(1, len(self.octavesAnalyzers), i + 1)

                candidates = oa.elements[ph]
                if ph != "kp_after_contrast_limitation":
                    candidates = Utils.adaptKeypoints(candidates, i - 1)
                    img = self.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
                else:
                    height, width, _ = self.originalPicture.shape
                    img = np.zeros((oa.octaveHeight, oa.octaveWidth))
                    for elt in candidates:
                        img[elt[0], elt[1]] = 1

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
            plt.subplot(1, len(self.octavesAnalyzers), i + 1)

            candidates = oa.finalKeypoints
            candidates = Utils.adaptKeypoints(candidates, i - 1)
            candidates = Utils.adaptSigmas(candidates, self.sigmas)

            img = self.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
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

        candidates = self.key_points
        img = self.showKeyPoints(copy.deepcopy(self.originalPicture), candidates)
        plt.imshow(img)
        plt.title("Keypoints - " + str(len(self.key_points)))

        # Affichage des points par octaves
        self.saveToFile("final")
        plt.clf()
        plt.cla()

        return fi + 1

    def generateCSVArrayAnalysis(self, fi):
        Log.debug("Génération du tableau d'analyse des points")

        header = "Octave;" \
                 "Nombre de point de l'image;" \
                 "Nombre de point de faibles constrates éliminés;" \
                 "Nombre total d'extremums détectés;" \
                 "Nombre de points d’arrêtes éliminés"

        lines = [header]

        for oa in self.octavesAnalyzers:
            elt = "\n" + str(oa.octaveNb) + ";"
            elt += str(oa.octaveWidth * oa.octaveHeight) + ";"
            elt += str(oa.octaveWidth * oa.octaveHeight - len(oa.elements["kp_after_contrast_limitation"])) + ";"
            elt += str(len(oa.elements["kp_after_extremum_detection"])) + ";"
            elt += str(len(oa.elements["kp_after_extremum_detection"]) - len(oa.elements["kp_after_hessian_filter"]))

            lines.append(elt)

        with open(self.outpath + "analyzer.csv", "w") as f:
            f.writelines(lines)

        return fi

    def generateAnalysisGraph(self, fi):
        Log.debug("Génération du graph d'analyse")

        plt.figure(fi + 1)

        X, Y = [], []

        for oa in self.octavesAnalyzers[::-1]:
            X.append(oa.resolution)
            Y.append(len(oa.finalKeypoints))

        plt.scatter(X, Y)
        self.saveToFile("kp_evolution")

        return fi + 1

    def saveMatrixKeypoints(self, fi):
        Log.debug("Génération de la matrice des points clés")
        matrix = np.matrix(self.key_points)
        np.save(self.outpath + "keypoints.npy", matrix)

        return fi

    def saveMatrixDescriptors(self, fi):
        Log.debug("Génération de la matrice des descripteurs")
        matrix = np.matrix(self.descriptors)
        np.save(self.outpath + "descriptors.npy", matrix)

        return fi

    def saveCv2Sift(self, fi):
        Log.debug("Génération de l'image de SIFT par OpenCV")
        gray = cv2.cvtColor(self.originalPicture, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        out_image = cv2.drawKeypoints(gray,
                                      kp,
                                      self.originalPicture,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.figure(fi + 1)
        plt.title("Keypoints - " + str(len(kp)))
        plt.imshow(out_image)
        self.saveToFile("cv2_result")

        return fi + 1
