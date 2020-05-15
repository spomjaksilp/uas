"""
From Paris
"""

import numpy as np
from scipy.spatial import distance
from uas import Lattice, Move, Plan
from . import StrategyTemplate


class French(StrategyTemplate):
    """
    calculate coordinates of state and target
    calculate distances
    sort distances
    calculate moves
    """

    def __init__(self, start: Lattice, target: Lattice, planner: Plan):
        self.start = start
        self.states = [start]
        self.target = target
        self.plan = planner()
        # first assert that a possible solution exists
        assert np.sum(start.value) <= np.sum(target.value), f"Unsolvable {np.sum(start.value)} sites cannot be sorted " \
                                                           f"to {np.sum(target.value)} sites"

    @property
    def current_state(self):
        return self.states[-1]

    def calculate_distances(self):
        dist = distance.cdist(self.start.coordinates, self.target.coordinates)
        return np.argsort(dist)

    def run(self):
        states_visited = []
        targets_visited = []
        for coord in self.calculate_distances():
            s_0 = self.start.coordinates[coord[0]]
            s_1 = self.target.coordinates[coord[1]]
            if self.current_state == self.target:
                # check for success
                break
            if s_1 in targets_visited:
                # already sorted
                continue
            if s_0 in states_visited:
                # there is no atom, go on
                continue
            self.states.append(self.plan.add_move(origin=s_0, target=s_1, lattice=self.current_state))
        return self.plan, self.current_state
