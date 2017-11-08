import cv2
import math
import numpy as np


class Image:
    @staticmethod
    def loadMatrix(path, **kwargs):
        flag = kwargs.get('flag', cv2.IMREAD_GRAYSCALE)
        data = cv2.imread(path, flag)

        return Image(data)

    @staticmethod
    def getOctave(image_matrix, octave):
        p = float(2 ** octave)
        return cv2.resize(image_matrix, (0, 0), fx=1.0 / p, fy=1.0 / p)

    @staticmethod
    def applyGaussianFilter(image_matrix, sigma):
        new_matrix = np.copy(image_matrix)
        height, width, channels = new_matrix.shape

        def gaussian(x, y):
            global sigma
            temp = math.exp(-((x ** 2) + (y ** 2)) / (2 * sigma ** 2))
            return temp / (2 * sigma * sigma * math.pi)

        for x in range(width):
            for y in range(height):
                new_matrix[x, y] = gaussian(x, y)

        return new_matrix

    @staticmethod
    def makeDifference(img1, img2):
        return img1 - img2


if __name__ == '__main__':
    pass
