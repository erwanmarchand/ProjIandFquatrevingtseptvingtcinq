from scipy import misc
import numpy as np
import cv2
import sys
import os
import math


def gaussian(x,y,sigma):
    temp = math.exp(-((x*x)+(y*y))/(2*sigma*sigma))
    return temp/(2*sigma*sigma*math.pi)

imageLena = cv2.imread('lena.jpg')
imageLenaGrey = cv2.cvtColor( imageLena, cv2.COLOR_RGB2GRAY )

cv2.imwrite( "grey.png", imageLenaGrey )

sigma = 1.6

for k in range (1,5):

    imageGreyBlur = imageLenaGrey.copy()

    maxDist = int(math.floor(3 * k * sigma)+1)

    gauss = np.zeros((maxDist, maxDist))

    for x in range(0,maxDist):
        for y in range(0, maxDist):
            gauss[x,y] = gaussian(x,y,sigma)


    for x in range(0, imageGreyBlur.shape[0]):
        for y in range(0, imageGreyBlur.shape[1]):
            pixel = 0
            for distancex in range(-maxDist+1, maxDist):
                for distancey in range(-maxDist+1, maxDist):
                    x2 = x + distancex
                    y2 = y + distancey
                    if (x2 >= 0 and y2 >=0 and x2 < imageGreyBlur.shape[0] and y2 < imageGreyBlur.shape[1] ):
                        pixel = pixel + (imageLenaGrey[x2,y2] * gauss[abs(distancex),abs(distancey)])
            imageGreyBlur[x,y] = pixel


    #print imageLenaGrey.shape[0]

    #print imageGreyBlur[100,100]

    
    cv2.imwrite( "blur.k="+str(k)+".png", imageGreyBlur )

print imageLena.shape
print imageLenaGrey.shape
print gaussian(1,1,1)
