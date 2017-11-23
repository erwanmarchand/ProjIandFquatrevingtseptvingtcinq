# -*- coding: utf-8 -*-


class OctaveAnalyzer:
    def __init__(self, octave_nb, width, height):
        self.octaveNb = octave_nb

        self.octaveWidth = width
        self.octaveHeight = height

        self.elements = {}
        self.finalKeypoints = []
