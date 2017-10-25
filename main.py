from scipy import misc
import numpy as np
import cv2
import sys
import os
import math
from scipy import signal

IMAGES_PATH = "Images\\"
ORIGINAL_IMAGES_PATH = IMAGES_PATH+"Original\\"
GENERATED_IMAGES_PATH = IMAGES_PATH+"Generated\\"

def gaussian(x,y,sigma):
    temp = math.exp(-((x*x)+(y*y))/(2*sigma*sigma))
    return temp/(2*sigma*sigma*math.pi)

imageLena = cv2.imread(ORIGINAL_IMAGES_PATH+'lena.jpg')
imageLenaGrey = cv2.cvtColor( imageLena, cv2.COLOR_RGB2GRAY )

#cv2.imwrite( GENERATED_IMAGES_PATH+"grey.png", imageLenaGrey )

sigma = 1.6
octave = 1
imageGreyBlur = []

for i in range (1,4):

    dim = (imageLenaGrey.shape[0]/octave,imageLenaGrey.shape[1]/octave)
    resized = cv2.resize(imageLenaGrey, dim)
    cv2.imwrite( GENERATED_IMAGES_PATH+"blur.k="+str(0)+".octave="+str(octave)+".png",resized )

    for k in range (1,6):

        imageGreyBlur.append(resized.copy())

        maxDist = int(math.floor(3 * k * sigma)+1)

        ## USING NP CONVOLUTION

        sizeKernel = 2*maxDist-1

        kernel = np.zeros( (sizeKernel,sizeKernel) )

        for x in range(0,sizeKernel):
            for y in range(0, sizeKernel):
                center = int(math.floor(sizeKernel/2))
                kernel[x,y] = gaussian(x-center,y-center,k*sigma)

        #print kernel

        imageGreyBlur[k-1] = signal.convolve2d(resized, kernel, boundary='symm', mode='same')


        ## HOMEMADE CONVOLUTION

    #    gauss = np.zeros((maxDist, maxDist))
    #
    #    for x in range(0,maxDist):
    #        for y in range(0, maxDist):
    #            gauss[x,y] = gaussian(x,y,k*sigma)
    #
    #
    #    for x in range(0, imageGreyBlur[k-1].shape[0]):
    #        for y in range(0, imageGreyBlur[k-1].shape[1]):
    #            pixel = 0
    #            for distancex in range(-maxDist+1, maxDist):
    #                for distancey in range(-maxDist+1, maxDist):
    #                    x2 = x + distancex
    #                    y2 = y + distancey
    #                    if (x2 >= 0 and y2 >=0 and x2 < imageGreyBlur[k-1].shape[0] and y2 < imageGreyBlur[k-1].shape[1] ):
    #                        pixel = pixel + (imageLenaGrey[x2,y2] * gauss[abs(distancex),abs(distancey)])
    #            imageGreyBlur[k-1][x,y] = pixel


        #print imageLenaGrey.shape[0]

        #print imageGreyBlur[100,100]

        print("sigma = "+str(sigma))
        print("octave = "+str(i))
        print("k = "+str(k))
        print("\n")
        
        cv2.imwrite( GENERATED_IMAGES_PATH+"blur.k="+str(k)+".octave="+str(octave)+".png", imageGreyBlur[k-1] )

    octave = octave * 2

#print imageLena.shape
#print imageLenaGrey.shape
#print gaussian(1,1,1)
