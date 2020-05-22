"""

"""

import numpy as np
from numba import types, typed, typeof, njit
from numba.experimental import jitclass
from uas.helper import ArrayMixin


spec_lattice = [
    ("array", types.boolean[:, :]),
    ("spacing", types.float32[:]),
    ("_coordinates_cached", types.boolean),
    ("_coordinates", types.int16[:, :]),
]


@njit
def calculate_coordinates(input_array):
    return np.argwhere(input_array).astype(np.int16)


# @jitclass(spec_lattice)
class Lattice(ArrayMixin):
    def __init__(self, array: np.ndarray, spacing: np.ndarray = np.array([5e-6, 5e-6], dtype=np.float32)):
        self._array = array
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

    def move(self, origin: np.ndarray, target: np.ndarray):
        self.value[tuple(origin)] = 0
        self.value[tuple(target)] = 1

    def discard(self, origin: np.ndarray):
        self.value[tuple(origin)] = 0


def lattice_from_lattice(lattice):
    obj = Lattice(array=np.zeros_like(lattice.value), spacing=lattice.spacing)
    obj._array = np.copy(lattice.value)
    return obj
