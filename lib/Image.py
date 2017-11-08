import cv2
import math
import numpy as np
import scipy.signal as signal


class Image:
    @staticmethod
    def loadMatrix(path, **kwargs):
        flag = kwargs.get('flag', cv2.IMREAD_GRAYSCALE)
        data = cv2.imread(path, flag)
        # data = data/float(np.max(data))

        return data

    @staticmethod
    def getOctave(image_matrix, octave):
        p = float(2 ** octave)

        return cv2.resize(image_matrix, (0, 0), fx=(1.0 / p), fy=(1.0 / p))

    @staticmethod
    def applyGaussianFilter(image_matrix, sigma):
        filter = Image.createGaussianFilter(3, sigma)
        return signal.convolve2d(image_matrix, filter)

    @staticmethod
    def createGaussianFilter(P, sigma):
        matrix = np.zeros((2 * P + 1, 2 * P + 1))
        som = 0
        for m in range(-P, P + 1):
            for n in range(-P, P + 1):
                matrix[m + P][n + P] = math.exp(-(n ** 2 + m ** 2) / (2 * (sigma ** 2)))
                som += matrix[m + P][n + P]
        matrix = matrix / som
        return matrix

    @staticmethod
    def makeDifference(img1, img2):
        return img1 - img2


if __name__ == '__main__':
    pass
