"""

"""

import logging
import pytest
import numpy as np
from copy import deepcopy
from uas import Lattice, Frontend
from uas.helper import Sample
from uas.strategy import StrategyTemplate, French

logging.basicConfig(level=logging.INFO)

STRATEGIES = [
    French
]

SHAPES = [
    (15, 15),
    (20, 20),
    (100, 100),
]

RECT_SIZES = [
    (8, 8),
    (10, 10)]


class TestStrategy:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_jit_compilation_time(self, strategy: StrategyTemplate):
        # just do a sorting example to get numba to compile things
        # assemble
        start_state = Lattice(np.load(f"./data/sample_10_10.npy"))

        sample = Sample((10, 10))
        sample.add_rect(origin=(0, 0), size=(5, 5))
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=deepcopy(start_state), target=target_state)
        plan, end_state = strat.run()
        timeline = Frontend(spacing=start_state.spacing, plan=plan).parse_plan()

    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_do_nothing(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[5, 5]])
        start_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=start_state)
        plan, end_state = strat.run()
        timeline = Frontend(spacing=start_state.spacing, plan=plan).parse_plan()

    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_one_straight(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[4, 5]])
        start_state = Lattice(sample.value)

        sample = Sample(shape)
        sample.add_position([[5, 5]])
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state)
        plan, end_state = strat.run()
        timeline = Frontend(spacing=start_state.spacing, plan=plan).parse_plan()

    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_one_straight_one_left(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[4, 5], [4, 4]])
        start_state = Lattice(sample.value)

        sample = Sample(shape)
        sample.add_position([[5, 5]])
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state)
        plan, end_state = strat.run()
        timeline = Frontend(spacing=start_state.spacing, plan=plan).parse_plan()

    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_two(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[1, 7], [1, 8]])
        start_state = Lattice(sample.value)

        sample = Sample(shape)
        sample.add_position([[5, 5], [5, 6]])
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state)
        plan, end_state = strat.run()
        timeline = Frontend(spacing=start_state.spacing, plan=plan).parse_plan()

    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("size", RECT_SIZES)
    def test_rect(self, strategy: StrategyTemplate, shape: tuple, size: tuple):
        # assemble
        start_state = Lattice(np.load(f"./data/sample_{shape[0]}_{shape[1]}.npy"))

        sample = Sample(shape)
        origin = np.array(shape) / 2
        # sample.add_rect(origin=tuple(origin.astype(np.int16)), size=size)
        sample.add_rect(origin=(2, 2), size=size)
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state)
        plan, end_state = strat.run()
        timeline = Frontend(spacing=start_state.spacing, plan=plan).parse_plan()
