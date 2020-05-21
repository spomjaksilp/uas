"""

"""

import numpy as np
from numba import types, typed, typeof
from numba.experimental import jitclass
from uas.helper import ArrayMixin


spec_lattice = [
    ("array", types.int16[:, :]),
    ("spacing", types.float32[:]),
    ("_coordinates_cached", types.boolean),
    ("_coordinates", types.int16[:, :]),
]


# @jitclass(spec_lattice)
class Lattice(ArrayMixin):
    def __init__(self, array: np.ndarray, spacing: list = [5e-6, 5e-6]):
        self.array = array
        self.spacing = spacing
        self._coordinates_cached = False
        self._coordinates = np.zeros((0, 0), dtype=np.int16)

    @property
    def coordinates(self):
        if self._coordinates_cached:
            return self._coordinates
        # get coordinates
        else:
            self._coordinates = np.array(np.where(self.array == 1), dtype=np.int16).T
            self._coordinates_cached = True
            return self._coordinates

    @classmethod
    def from_lattice(cls, lattice):
        obj = cls(array=np.zeros_like(lattice.value), spacing=lattice.spacing)
        obj.array = np.copy(lattice.array)
        return obj

    @staticmethod
    def move(lattice, origin: np.ndarray, target: np.ndarray or None):
        new_lattice = Lattice.from_lattice(lattice)
        new_lattice.value[tuple(origin)] = 0
        if target is not None:
            new_lattice.value[tuple(target)] = 1
        return new_lattice
