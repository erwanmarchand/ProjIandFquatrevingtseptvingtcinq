import math

import cv2
import numpy as np
from scipy import signal

from lib.ExtremaDetector import ExtremaDetector

IMAGES_PATH = "images\\"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original\\"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated\\"

NAME_PICTURE = 'lena2'
sigma = 1.6
octave = 1


maxI = 3
maxk = 5


extremaDetect = ExtremaDetector(imageLenaGrey,sigma,maxk,maxI)
extremaDetect.generateDiffGaussian()


for i in range (1,maxI+1):

    imageGreyBlurOctave = []
    diffGaussOctave = []

    dim = (imageLenaGrey.shape[1]/octave,imageLenaGrey.shape[0]/octave)
    resized = cv2.resize(imageLenaGrey, dim)
    imageGreyBlurOctave.append(resized.copy())
    cv2.imwrite( GENERATED_IMAGES_PATH+NAME_PICTURE+".blur.k="+str(0)+".octave="+str(octave)+".png",resized )

    for k in range (1,maxk+1):

        imageGreyBlurOctave.append(resized.copy())

        maxDist = int(math.floor(3 * k * sigma)+1)

        ## USING NP CONVOLUTION

        sizeKernel = 2*maxDist-1

        kernel = np.zeros( (sizeKernel,sizeKernel) )

        for x in range(0,sizeKernel):
            for y in range(0, sizeKernel):
                center = int(math.floor(sizeKernel/2))
                kernel[x,y] = gaussian(x-center,y-center,k*sigma)

        #print kernel

        imageGreyBlurOctave[k] = signal.convolve2d(resized, kernel, boundary='symm', mode='same')

        print("sigma = "+str(sigma))
        print("octave = "+str(i))
        print("k = "+str(k))
        print("\n")
        
        cv2.imwrite( GENERATED_IMAGES_PATH+NAME_PICTURE+".blur.k="+str(k)+".octave="+str(octave)+".png", imageGreyBlurOctave[k] )

        

        #difference de gaussien
        diffGaussOctave.append(resized.copy())
        for x in range(0, diffGaussOctave[k-1].shape[0]):
            for y in range(0, diffGaussOctave[k-1].shape[1]):
                diffGaussOctave[k-1][x,y] = max(0,imageGreyBlurOctave[k][x,y] - imageGreyBlurOctave[k-1][x,y])

        cv2.imwrite( GENERATED_IMAGES_PATH+NAME_PICTURE+".diffgaussien.k="+str(k-1)+".octave="+str(octave)+".png", diffGaussOctave[k-1] )

    imageGreyBlur.append(imageGreyBlurOctave)
    diffGauss.append(diffGaussOctave)

    octave = octave * 2

#print imageLena.shape
#print imageLenaGrey.shape
#print gaussian(1,1,1)
