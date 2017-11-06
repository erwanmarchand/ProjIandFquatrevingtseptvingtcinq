from Point import *

class IPoint(Point):
    def __init__(self, x, y, sigma):
        Point.__init__(self, x, y)
        self.sigma = sigma
