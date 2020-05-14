"""

"""
from .builder import Sample


class ArrayMixin:
    """
    Enables to use == on Sample and Lattice
    """
    def __eq__(self, other):
        from numpy import array_equal
        return array_equal(self, other)
