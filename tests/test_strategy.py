import pytest
from uas import Lattice
from uas.helper import Sample
from uas.strategy import French
from uas import Plan


class TestStrategy:
    SHAPE = [100, 100]

    def test_do_nothing(self):
        # assemble
        sample = Sample(self.SHAPE)
        sample.add_position([[50, 50]])
        start_state = Lattice(sample.value)

        # act
        strat = French(start=start_state, target=start_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        assert len(plan) == 0
        assert end_state == start_state

    def test_one_straight(self):
        # assemble
        sample = Sample(self.SHAPE)
        sample.add_position([[40, 50]])
        start_state = Lattice(sample.value)

        sample = Sample(self.SHAPE)
        sample.add_position([[50, 50]])
        target_state = Lattice(sample.value)

        # act
        strat = French(start=start_state, target=target_state, planner=Plan)
        plan, end_state = strat.run()

        # assert
        assert len(plan) == 1
        assert target_state == start_state

