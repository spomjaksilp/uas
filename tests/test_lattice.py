import pytest
from uas.helper import Sample
from uas import Lattice


class TestLattice:
    SHAPE = [100, 100]

    def test_init_lattice(self):
        # assemble
        sample = Sample(self.SHAPE)
        sample.add_position([[50, 50]])

        # act
        lattice = Lattice(sample.value)

        # assert
        assert sample.value == lattice.value
