from scipy import misc
import numpy as np
import cv2
import sys
import os
import math
from scipy import signal
from pointdInteret import PointdInteret

class ExtremaDetector:

    def __init__(self, imageInputGrey, sigma, kMax, nbrOctave):
        self.imageInputGrey = imageInputGrey
        self.sigma = sigma
        self.kMax = kMax
        self.nbrOctave = nbrOctave
        self.imageGreyBlur = []
        self.diffGauss = []

    def gaussian(self, x,y,sigma):
        temp = math.exp(-((x*x)+(y*y))/(2*sigma*sigma))
        return temp/(2*sigma*sigma*math.pi)

    def generateDiffGaussian(self):

        octave = 1

        for i in range (1,self.nbrOctave+1):

            imageGreyBlurOctave = []
            diffGaussOctave = []

            dim = (self.imageInputGrey.shape[1]/octave,self.imageInputGrey.shape[0]/octave)
            resized = cv2.resize(self.imageInputGrey, dim)
            imageGreyBlurOctave.append(resized.copy())
            #cv2.imwrite( GENERATED_IMAGES_PATH+NAME_PICTURE+".blur.k="+str(0)+".octave="+str(octave)+".png",resized )

            for k in range (1,self.kMax+1):

                imageGreyBlurOctave.append(resized.copy())

                maxDist = int(math.floor(3 * k * self.sigma)+1)

                ## USING NP CONVOLUTION

                sizeKernel = 2*maxDist-1

                kernel = np.zeros( (sizeKernel,sizeKernel) )

                for x in range(0,sizeKernel):
                    for y in range(0, sizeKernel):
                        center = int(math.floor(sizeKernel/2))
                        kernel[x,y] = self.gaussian(x-center,y-center,k*self.sigma)

                #print kernel

                imageGreyBlurOctave[k] = signal.convolve2d(resized, kernel, boundary='symm', mode='same')

                print("sigma = "+str(self.sigma))
                print("octave = "+str(i))
                print("k = "+str(k))
                print("\n")
                
                #cv2.imwrite( GENERATED_IMAGES_PATH+NAME_PICTURE+".blur.k="+str(k)+".octave="+str(octave)+".png", imageGreyBlurOctave[k] )

                

                #difference de gaussien
                diffGaussOctave.append(resized.copy())
                for x in range(0, diffGaussOctave[k-1].shape[0]):
                    for y in range(0, diffGaussOctave[k-1].shape[1]):
                        diffGaussOctave[k-1][x,y] = max(0,imageGreyBlurOctave[k][x,y] - imageGreyBlurOctave[k-1][x,y])

                #cv2.imwrite( GENERATED_IMAGES_PATH+NAME_PICTURE+".diffgaussien.k="+str(k-1)+".octave="+str(octave)+".png", diffGaussOctave[k-1] )

            self.imageGreyBlur.append(imageGreyBlurOctave)
            self.diffGauss.append(diffGaussOctave)

            octave = octave * 2

    def createPicturesGaussian(self, PATH) :
        #TODO
        return