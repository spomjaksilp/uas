"""
Build a numpy grid with sample configurations
"""

import numpy as np
from skimage.draw import disk


def generate_mask_box(origin, size, shape):
    """
    generate a box of ones in a 2d image (with size shape) filled with zeros as mask.
    mask center is denoted by center.
    the size is the half of the size in one direction: e.g. size = 2 means [0, 1, 1, center, 1, 1, 0]

    :param center: tuple/array with coordinates (x, y)
    :param size: integer
    :param shape: tuple/array like (size_x, size_y)
    """
    size_x, size_y = [int(_) for _ in size]
    mask = np.zeros(shape)
    x, y = np.array(origin)

    x_low = x
    x_high = x + size_x
    y_low = y
    y_high = y + size_y

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
        result = array_equal(self.value, other.value)
        return result

    @property
    def value(self):
        """
        :return: (ndarray) returns the array
        """
        return self.array


class Sample(ArrayMixin):
    def __init__(self, shape: tuple = (100, 100)):
        """
        :param shape: list as used in numpy
        """
        self.array = np.zeros(shape, dtype=np.bool)

    @property
    def shape(self):
        """

        :return: (list) shape
        """
        return self.array.shape

    def add_position(self, positions: tuple = ((10, 20), (50, 50))):
        """

        :param positions: list of sites
        :return:
        """
        for p in positions:
            self.array[tuple(p)] = 1

    def add_rect(self, origin: tuple = (20, 20), size: tuple = (10, 10)):
        """

        :param origin: (list) origin coordinate
        :param size: (list) size
        :return:
        """
        self.array += generate_mask_box(origin=origin, size=size, shape=self.shape).astype(np.bool)

    def add_disk(self, origin: tuple = (20, 20), radius: float = 10):
        """

        :param origin: (list) origin coordinate
        :param radius: (int) radius
        :return:
        """
        self.array += disk(center=origin, radius=radius, shape=self.shape)

    def add_random(self, probability: (float or np.ndarray) = 0.5):
        """

        :param probability: (float of ndarray) probability of loading sites, site selective if ndarray
        :return:
        """
        if type(probability) is np.ndarray:
            assert np.array_equal(probability.shape, self.array.shape), f"Probability shape missmatch:" \
                                                                         f"{probability.shape}"
        self.array = np.where(np.random.random(self.shape) < probability, np.ones_like(self.array), self.array)
