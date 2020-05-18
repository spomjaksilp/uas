import logging
import pytest
import numpy as np
from uas import Lattice
from uas.helper import Sample
from uas.strategy import StrategyTemplate, French
from uas import Plan

logging.basicConfig(level=logging.DEBUG)

TO_TEST = [
    [French, (100, 100)],
]


class TestStrategy:
    @pytest.mark.parametrize("strategy, shape", TO_TEST)
    def test_do_nothing(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[50, 50]])
        start_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=start_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        assert len(plan) == 0
        assert end_state == start_state

    @pytest.mark.parametrize("strategy, shape", TO_TEST)
    def test_unsolvable(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        start_state = Lattice(sample.value)
        target = Sample(shape)
        target.add_position([[50, 50]])
        target_state = Lattice(target.value)

        # act & assert
        with pytest.raises(AssertionError, match=r"^Unsolvable.*"):
            strat = strategy(start=start_state, target=target_state, planner=Plan)
            plan, end_state = strat.run()

    @pytest.mark.parametrize("strategy, shape", TO_TEST)
    def test_one_straight(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[40, 50]])
        start_state = Lattice(sample.value)

        sample = Sample(shape)
        sample.add_position([[50, 50]])
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        assert target_state == end_state

    @pytest.mark.parametrize("strategy, shape", TO_TEST)
    def test_one_straight_one_left(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[40, 50], [45, 50]])
        start_state = Lattice(sample.value)

        sample = Sample(shape)
        sample.add_position([[50, 50]])
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        assert len(plan) == 2
        assert target_state == end_state

    @pytest.mark.parametrize("strategy, shape", TO_TEST)
    def test_two(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        sample = Sample(shape)
        sample.add_position([[40, 50], [50, 70]])
        start_state = Lattice(sample.value)

        sample = Sample(shape)
        sample.add_position([[50, 50], [50, 60]])
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        assert len(plan) == 2
        assert target_state == end_state

    @pytest.mark.parametrize("strategy, shape", TO_TEST)
    def test_rect(self, strategy: StrategyTemplate, shape: tuple):
        # assemble
        start_state = Lattice(np.load("./data/sample_100_100.npy"))

        sample = Sample(shape)
        sample.add_rect(origin=(20, 20), size=(5, 5))
        target_state = Lattice(sample.value)

        # act
        strat = strategy(start=start_state, target=target_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        logging.debug(strat.report)
        assert target_state == end_state
