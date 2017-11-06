import numpy as np
import cv2
from scipy import signal
import math


class Image:
    @staticmethod
    def from_path(path, **kwargs):
        flag = kwargs.get('flag', cv2.IMREAD_GRAYSCALE)

        data = cv2.imread(path, flag)

        return Image(data)

    def __init__(self, data, **kwargs):
        self.image = data
        self.height, self.width, self.channels = self.image.shape

        self.octave = kwargs.get('octave', 0)

    def __getitem__(self, key):
        return self.image[key]

    def get_next_octave(self):
        nI = cv2.resize(self.image, (0, 0), fx=0.5, fy=0.5)

        return Image(nI, octave=self.octave + 1)

    def apply_gaussian_filter(self, sigma):
        def gaussian(self, x, y):
            global sigma
            temp = math.exp(-((x * x) + (y * y)) / (2 * sigma * sigma))
            return temp / (2 * sigma * sigma * math.pi)

        for x in range(self.width)
            for y in range(self.height):
                self.image[x][y] = gaussian(x, y, sigma)
