"""

"""

import numpy as np
from numba import types, typed, typeof, njit
from numba.experimental import jitclass
from uas.helper import ArrayMixin
from uas.helper import type_site_matrix, type_coordinate_matrix


spec_lattice = [
    ("array", types.boolean[:, :]),
    ("spacing", types.float32[:]),
    ("_coordinates_cached", types.boolean),
    ("_coordinates", types.int16[:, :]),
]


@njit
def calculate_coordinates(input_array):
    return np.argwhere(input_array).astype(np.int16)


@njit(type_site_matrix(type_coordinate_matrix, type_site_matrix), fastmath=True)
def array_from_coordinates(coordinates, new_array):
    for coord in coordinates:
        for _c, _s in zip(coord, new_array.shape):
            if _c > _s:
                _c = _c - 1
        new_array[coord[0], coord[1]] = 1
    return new_array


# @jitclass(spec_lattice)
class Lattice(ArrayMixin):
    def __init__(self, array: np.ndarray, spacing: np.ndarray = np.array([5e-6, 5e-6], dtype=np.float32)):
        """

        :param array:
        :param spacing: (ndarray) [µm]
        """
        self._array = array
        self._array_cached = True
        self._shape = self._array.shape
        self.spacing = spacing
        self._coordinates_cached = False
        self._coordinates = np.zeros((2, 2), dtype=np.int16)

    @property
    def coordinates(self):
        if self._coordinates_cached:
            return self._coordinates
        # get coordinates
        else:
            self._coordinates = calculate_coordinates(self.value)
            self._coordinates_cached = True
            return self._coordinates

    @property
    def value(self):
        """
        :return: (ndarray) returns the array
        """
        if self._array_cached:
            return self._array
        # get array
        else:
            self._array = array_from_coordinates(self.coordinates, np.zeros(self._shape, dtype=np.bool))
            self._array_cached = True
            return self._array

    def set_coordinates(self, coordinates: np.ndarray):
        """

        :param coordinates: ndarray
        :return:
        """
        self._coordinates = coordinates
        self._coordinates_cached = True

    def set_value(self, value: np.ndarray):
        """

        :param value: ndarray
        :return:
        """
        self._array = value
        self._array_cached = True

    def resample(self, factor: np.ndarray):
        """
        Resample the lattice. The underlying array is then reconstructed
        from coordinates.
        :param factor:
        :return:
        """
        new_coordinates = np.copy(self.coordinates)
        new_coordinates[:, 0] = np.around(new_coordinates[:, 0] * factor[0], decimals=0).astype(np.int16)
        new_coordinates[:, 1] = np.around(new_coordinates[:, 1] * factor[1], decimals=0).astype(np.int16)
        new_shape = np.around(np.array(self._array.shape) * factor, decimals=0).astype(np.int16)
        self.set_coordinates(new_coordinates)
        self.set_value(array_from_coordinates(new_coordinates, np.zeros(new_shape, dtype=np.bool)))
        self._shape = new_shape
        self.spacing = self.spacing / factor

    def rescale(self, spacing: np.ndarray):
        """
        Rescale the lattice to a new spacing. The underlying array is the reconstructed.
        :param spacing: (ndarray)
        :return:
        """
        factor = self.spacing / spacing
        self.resample(factor)

    def move(self, origin: np.ndarray, target: np.ndarray):
        self.value[tuple(origin)] = 0
        self.value[tuple(target)] = 1
        self._coordinates_cached = False

    def discard(self, origin: np.ndarray):
        self.value[tuple(origin)] = 0
        self._coordinates_cached = False


def lattice_from_lattice(lattice):
    obj = Lattice(array=np.zeros_like(lattice.value), spacing=lattice.spacing)
    obj._array = np.copy(lattice.value)
    return obj

#
# class Timeline:
