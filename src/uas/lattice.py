"""

"""

import numpy as np
from uas.helper import ArrayMixin


class Lattice(ArrayMixin):
    def __init__(self, array: np.ndarray, spacing: list = [5e-6, 5e-6]):
        self._array = array
        self.spacing = spacing
        self._coordinates = None

    @property
    def coordinates(self):
        if self._coordinates is None:
            # get coordinates
            self._coordinates = np.where(self._array == 1)
        return self._coordinates

    @classmethod
    def from_lattice(cls, lattice):
        obj = cls(array=np.zeros_like(lattice.value), spacing=lattice.spacing)
        obj._array = lattice._array
        return obj

    @staticmethod
    def move(lattice, origin: list, target: list):
        new_lattice = Lattice.from_lattice(lattice)
        new_lattice.value[origin] = 0
        new_lattice.value[target] = 1
        return new_lattice
