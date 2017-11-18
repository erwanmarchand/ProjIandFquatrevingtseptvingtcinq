# -*- coding: utf-8 -*-
from lib.ImageManager import *
import cv2
import numpy as np

IMAGES_PATH = "images/"
ORIGINAL_IMAGES_PATH = IMAGES_PATH + "original/"
GENERATED_IMAGES_PATH = IMAGES_PATH + "generated/"
DEBUG = True
SHOW_IMAGE = False
NAME_PICTURE = 'lena.jpg'

s = 3
octave = 3

# On charge l'image et on la redimensionne
img = ImageManager.loadMatrix(GENERATED_IMAGES_PATH+"DoG_0_0.jpg")

x_grad = np.gradient(img) 
hessian = np.empty((img.ndim, img.ndim) + img.shape, dtype=img.dtype) 
for k, grad_k in enumerate(x_grad):
    # iterate over dimensions
    # apply gradient again to every component of the first derivative.
    tmp_grad = np.gradient(grad_k) 
    for l, grad_kl in enumerate(tmp_grad):
        hessian[k, l, :, :] = grad_kl

hessian2 = np.empty((img.shape[0]-2,img.shape[1]-2), dtype=img.dtype)

for x in range(0, hessian2.shape[0]):
    for y in range(0, hessian2.shape[1]):
        hessian2[x][y] = img[x+2][y+1] + img[x][y+1] - (2*img[x+1][y+1])

test1= hessian[0][0]
test2= hessian[1][1]

for x in range(0, hessian2.shape[0]):
    for y in range(0, hessian2.shape[1]):
        test1[x+1][y+1] = test1[x+1][y+1] - hessian2[x][y]
        test2[x+1][y+1] = test2[x+1][y+1] - hessian2[x][y]

ImageManager.showImage(hessian[0][0], cmap='gray')
#ImageManager.showImage(hessian[0][1], cmap='gray')
#ImageManager.showImage(hessian[1][0], cmap='gray')
#ImageManager.showImage(hessian[1][1], cmap='gray')
ImageManager.showImage(test1, cmap='gray')
ImageManager.showImage(test2, cmap='gray')