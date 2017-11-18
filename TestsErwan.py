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

ImageManager.showImage(hessian[0][0], cmap='gray')
ImageManager.showImage(hessian[0][1], cmap='gray')
ImageManager.showImage(hessian[1][0], cmap='gray')
ImageManager.showImage(hessian[1][1], cmap='gray')