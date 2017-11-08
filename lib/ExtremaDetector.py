import numpy as np
import cv2
import math
from scipy import signal
import math

import cv2
import numpy as np
from scipy import signal


class ExtremaDetector:
    @staticmethod
    def differenceDeGaussienne(image, s, nb_octave):
        # Construction de la pyramide de gaussiennes


