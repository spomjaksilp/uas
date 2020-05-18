"""

"""

import numpy as np
from uas.helper import ArrayMixin


class Lattice(ArrayMixin):
    def __init__(self, array: np.ndarray, spacing: list = [5e-6, 5e-6]):
        self.array = array
        self.spacing = spacing
        self._coordinates = None

    @property
    def coordinates(self):
        if self._coordinates is None:
            # get coordinates
            self._coordinates = np.array(np.where(self.array == 1)).T
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
