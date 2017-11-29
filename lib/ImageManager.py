# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

from lib.Filter import *


class ImageManager:
    def __init__(self):
        pass

    @staticmethod
    def loadMatrix(path, *args):
        data = cv2.imread(path, *args)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        return data

    @staticmethod
    def writeImage(image, path):
        cv2.imwrite(path, image)

    @staticmethod
    def normalizeImage(image):
        image_max = max(abs(np.max(image)), abs(np.min(image)))
        if image_max > 1:  # On vérifie que l'image n'est pas déja normalisée
            return image / image_max
        else:
            return image

    @staticmethod
    def divideSizeBy2(image):
        return ImageManager.getOctave(image, 1)

    @staticmethod
    def getGreyscale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def getOctave(image_matrix, octave):
        p = float(2 ** octave)

        return cv2.resize(image_matrix, (0, 0), fx=(1.0 / p), fy=(1.0 / p))

    @staticmethod
    def rotate(image, angle, center=None):
        """
        Effectue la rotation d'une image
        :param image: Image a tourner
        :param angle: Angle en degrés
        :param center: Centre de rotation (row, col)
        :return:
        """

        rows, cols = ImageManager.getDimensions(image)

        if center is not None:
            center_row, center_col = center
        else:
            center_row, center_col = (int(rows / 2), int(cols / 2))

        m = cv2.getRotationMatrix2D((center_col, center_row), angle, 1)
        return cv2.warpAffine(image, m, (cols, rows))

    @staticmethod
    def applyGaussianFilter(image, sigma):
        img_filtered = Filter.createGaussianFilter(int(sigma * 3), sigma)
        return Filter.convolve2D(image, img_filtered)

    @staticmethod
    def showKeyPoint(image, keypoint, **kwargs):
        rows, cols = ImageManager.getDimensions(image)
        point_color = [int(keypoint[2] * 1000) % 256, int(keypoint[2] * 333) % 256, int(keypoint[2] * 666) % 256]

        # On détermine si l'argument "keypoint" est un descripteur complet ou juste un point clé
        if len(keypoint) <= 4:
            try:
                kp3 = keypoint[3]
            except IndexError:
                kp3 = None

            if kp3 is not None:
                radius = (keypoint[2] ** 1.5) * 4 * min(cols, rows) / 1024
                image = cv2.circle(image, (keypoint[1], keypoint[0]), int(radius), point_color, 2)

                # On dessine la fleche représentant l'orientation du point clé
                image = cv2.arrowedLine(image,
                                 (int(keypoint[1]), int(keypoint[0])),
                                 (int(keypoint[1]) + int(np.cos(keypoint[3]) * radius),
                                  int(keypoint[0]) - int(np.sin(keypoint[3]) * radius)),
                                 point_color, 2)
            else:
                image = cv2.circle(image, (int(keypoint[1]), int(keypoint[0])), int(keypoint[2]) + 1, point_color, 5)
        else:
            radius = 10 * min(cols, rows) / 1024
            image = cv2.circle(image, (int(keypoint[1]), int(keypoint[0])), int(radius), point_color, 2)

        return image

    @staticmethod
    def makeDifference(img1, img2):
        return img1 - img2

    @staticmethod
    def showImage(image, **kwargs):
        plt.imshow(image, **kwargs)
        plt.show()

    @staticmethod
    def getDimensions(image):
        rows, cols = image.shape[0], image.shape[1]

        return rows, cols


if __name__ == '__main__':
    pass
