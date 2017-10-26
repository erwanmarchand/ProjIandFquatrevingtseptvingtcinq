from scipy import misc
import numpy as np
import cv2
import sys
import os
import math
from scipy import signal

class PointdInteret:

    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
