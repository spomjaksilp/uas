"""
Build a numpy grid with sample configurations
"""

import numpy as np
from skimage.draw import disk

from . import ArrayMixin


def generate_mask_box(center, size, shape):
    """
    generate a box of ones in a 2d image (with size shape) filled with zeros as mask.
    mask center is denoted by center.
    the size is the half of the size in one direction: e.g. size = 2 means [0, 1, 1, center, 1, 1, 0]

    :param center: tuple/array with coordinates (x, y)
    :param size: integer
    :param shape: tuple/array like (size_x, size_y)
    """
    size = int(size)
    mask = np.zeros(shape)
    x, y = np.array(center)

    x_low = x - size
    x_high = x + size + 1
    y_low = y - size
    y_high = y + size + 1

    # now catch all error cases (index to high, index negative ...)
    if x_low < 0:
        x_low = 0
    if y_low < 0:
        y_low = 0

    # >= because arrays start at 0
    if x_high > shape[0]:
        x_high = shape[0]
    if y_high > shape[1]:
        y_high = shape[1]

    mask[x_low:x_high, y_low:y_high] = 1
    return mask


class Sample(ArrayMixin):
    def __init__(self, shape=[100, 100]):
        """
        :param shape: list as used in numpy
        """
        self._grid = np.zeros(shape, dtype=np.bool)

    @property
    def shape(self):
        """

        :return: (list) shape
        """
        return self._grid.shape

    def add_position(self, positions=[[10, 20], [50, 50]]):
        """

        :param positions: list of sites
        :return:
        """
        for p in positions:
            self._grid[p] = 1

    def add_rect(self, origin=[20, 20], size=[10, 10]):
        """

        :param origin: (list) origin coordinate
        :param size: (list) size
        :return:
        """
        self._grid += generate_mask_box(center=origin, size=size, shape=self.shape)

    def add_disk(self, origin=[20, 20], radius=10):
        """

        :param origin: (list) origin coordinate
        :param radius: (int) radius
        :return:
        """
        self._grid += disk(center=origin, radius=radius, shape=self.shape)

    def add_random(self, probability=0.5):
        """

        :param probability: (float) probability of loading sites
        :return:
        """
        np.where(np.random.random(self.shape) < 0.5, np.ones_like(self._grid), self._grid)

    @property
    def value(self):
        """

        :return: (ndarray) returns the sample
        """
        return self._grid
