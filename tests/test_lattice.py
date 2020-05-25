import pytest
import numpy as np
from uas.helper import Sample
from uas import Lattice

SHAPES = [
    (15, 15),
    (20, 20),
    (100, 100),
    (1000, 1000)
]


class TestLattice:
    def test_jit_compilation_time(self):
        # assemble
        old_spacing = np.array((5e-6, 5e-6))
        factor = np.array((2, 2))
        sample = Sample((15, 15))
        sample.add_random(0.5)
        lattice = Lattice(sample.value, spacing=old_spacing)

        # act
        lattice.resample(factor)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_init_lattice(self, shape):
        # assemble
        sample = Sample(shape)
        sample.add_position(((1, 1), ))

        # act
        lattice = Lattice(sample.value)

        # assert
        assert sample == lattice

    @pytest.mark.parametrize("shape", SHAPES)
    def test_resample(self, shape):
        # assemble
        old_spacing = np.array((5e-6, 5e-6))
        factor = np.array((2, 2))
        sample = Sample(shape)
        sample.add_random(0.5)
        orig_lattice = Lattice(np.copy(sample.value), spacing=old_spacing)
        lattice = Lattice(sample.value, spacing=old_spacing)

        # act
        lattice.resample(factor)
        lattice.resample(1 / factor)

        # assert
        assert (orig_lattice.spacing == lattice.spacing).all()
        assert orig_lattice == lattice

    @pytest.mark.parametrize("shape", SHAPES)
    def test_rescale(self, shape):
        # assemble
        old_spacing = np.array((5e-6, 5e-6))
        new_spacing = old_spacing / 2
        sample = Sample(shape)
        sample.add_random(0.5)
        orig_lattice = Lattice(np.copy(sample.value), spacing=old_spacing)
        lattice = Lattice(sample.value, spacing=old_spacing)

        # act
        lattice.rescale(new_spacing)
        lattice.rescale(old_spacing)

        # assert
        assert (orig_lattice.spacing == lattice.spacing).all()
        assert orig_lattice == lattice

