# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from lib.debug.Log import *


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def showPyramid(pyramid, sigmas, **kwargs):
        """
        Open a window and show a Pyramid
        :param pyramid:     The pyramid we want to watch
        :param sigmas:      The different sigmas of the pyramid
        """
        nb_octave, nb_per_row = len(pyramid), len(pyramid[0])

        # Show pyramid
        for o in range(nb_octave):
            for k in range(nb_per_row):
                plt.subplot(nb_octave, nb_per_row, (k + 1) + o * nb_per_row)
                plt.title(str(o + 1) + " || " + str(round(sigmas[k], 4)))
                plt.imshow(pyramid[o][k], cmap=kwargs.get("cmap", 'gray'))
