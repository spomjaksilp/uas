"""
From Paris
"""

import numpy as np
from scipy.spatial import distance
from uas import Lattice, Plan, Type2
from . import StrategyTemplate


class French(StrategyTemplate):
    """
    src: Sylvain de Léséleuc (2018) - Quantum simulation of spin models with assembled arrays of Rydberg atoms, pp. 51
    * calculate coordinates of state and target
    * calculate distances
    * sort distances
    * calculate moves
    """

    def __init__(self, start: Lattice, target: Lattice, planner: Plan):
        self.start = start
        self.states = [start]
        self.target = target
        self.plan = planner(Type2)
        # first assert that a possible solution exists
        assert np.sum(start.value) >= np.sum(target.value), f"Unsolvable {np.sum(start.value)} sites cannot be sorted" \
                                                            f" to {np.sum(target.value)} sites"

    @property
    def current_state(self):
        return self.states[-1]

    def calculate_distances(self):
        dist = distance.cdist(self.current_state.coordinates, self.target.coordinates)
        dist_sorted = np.argsort(dist, axis=None)
        # https://stackoverflow.com/questions/29734660/python-numpy-keep-a-list-of-indices-of-a-sorted-2d-array
        coordinates_sorted = np.vstack(np.unravel_index(dist_sorted, dist.shape)).T
        return coordinates_sorted

    def run(self):
        # remove already sorted sites
        states_visited = []
        targets_visited = []
        for _already_sorted in np.array(np.where(self.start.value & self.target.value)).T:
            states_visited.append(_already_sorted)
            targets_visited.append(_already_sorted)
        # loop over sorted distances between start and target
        for coord in self.calculate_distances():
            # TODO use reshape to get rid of some part of the for loop
            s_0 = self.start.coordinates[coord[0]]
            s_1 = self.target.coordinates[coord[1]]
            if self.current_state == self.target:
                # check for success
                break
            if (s_1 in targets_visited) or (s_0 in states_visited):
                # already sorted or there is no atom, go on
                continue
            # add a move to the plan
            self.states.append(self.plan.add_move(origin=s_0, target=s_1, lattice=self.current_state))
            # mark sites as visited
            states_visited.append(s_0)
            targets_visited.append(s_1)
        return self.plan, self.current_state
