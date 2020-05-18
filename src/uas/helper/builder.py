"""
Build a numpy grid with sample configurations
"""

import numpy as np
from skimage.draw import disk


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


class ArrayMixin:
    """
    Enables to use == on Sample and Lattice
    """
    def __eq__(self, other):
        from numpy import array_equal
        return array_equal(self.value, other.value)

    @property
    def value(self):
        """
        :return: (ndarray) returns the array
        """
        return self._array


class Sample(ArrayMixin):
    def __init__(self, shape: list = [100, 100]):
        """
        :param shape: list as used in numpy
        """
        self._array = np.zeros(shape, dtype=np.bool)

    @property
    def shape(self):
        """

        :return: (list) shape
        """
        return self._array.shape

    def add_position(self, positions: list = [[10, 20], [50, 50]]):
        """

        :param positions: list of sites
        :return:
        """
        for p in positions:
            self._array[tuple(p)] = 1

    def add_rect(self, origin: list = [20, 20], size: list = [10, 10]):
        """

        :param origin: (list) origin coordinate
        :param size: (list) size
        :return:
        """
        self._array += generate_mask_box(center=origin, size=size, shape=self.shape)

    def add_disk(self, origin: list = [20, 20], radius: float = 10):
        """

        :param origin: (list) origin coordinate
        :param radius: (int) radius
        :return:
        """
        self._array += disk(center=origin, radius=radius, shape=self.shape)

    def add_random(self, probability: (float or np.ndarray) = 0.5):
        """

        :param probability: (float of ndarray) probability of loading sites, site selective if ndarray
        :return:
        """
        if type(probability) is np.ndarray:
            assert np.array_equal(probability.shape, self._array.shape), f"Probability shape missmatch:" \
                                                                         f"{probability.shape}"
        np.where(np.random.random(self.shape) < probability, np.ones_like(self._array), self._array)
