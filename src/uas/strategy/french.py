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
        self.current_state = start
        self.target = target
        self.plan = planner(Type2)
        self.report = {}
        # first assert that a possible solution exists
        assert np.sum(start.value) >= np.sum(target.value), f"Unsolvable {np.sum(start.value)} sites cannot be sorted" \
                                                            f" to {np.sum(target.value)} sites"

    def calculate_distances(self):
        dist = distance.cdist(self.start.coordinates, self.target.coordinates)
        dist_sorted = np.argsort(dist, axis=None)
        # https://stackoverflow.com/questions/29734660/python-numpy-keep-a-list-of-indices-of-a-sorted-2d-array
        coordinates_sorted = np.vstack(np.unravel_index(dist_sorted, dist.shape)).T
        return coordinates_sorted

    def run(self):
        # mark already sorted sites
        states_visited = np.where(self.start.value & self.target.value, np.ones_like(self.start.value),
                                  np.zeros_like(np.ones_like(self.start.value)))
        targets_visited = np.copy(states_visited)
        # loop over sorted distances between start and target
        for coord in self.calculate_distances():
            # TODO maybe use reshape to get rid of some part of the for loop
            s_0 = self.start.coordinates[coord[0]]
            s_1 = self.target.coordinates[coord[1]]
            if states_visited[tuple(s_0)] or targets_visited[tuple(s_1)]:
                # already sorted or there is no atom, go on
                continue
            # add a move to the plan
            self.current_state = self.plan.add_move(origin=s_0, target=s_1, lattice=self.current_state)
            # mark sites as visited
            states_visited[tuple(s_0)] = 1
            targets_visited[tuple(s_1)] = 1
        # drop left over atoms
        remainder = Lattice(np.logical_and(self.current_state.value, np.logical_not(self.target.value)),
                            spacing=self.current_state.spacing)
        self.report.update({"site-moves": len(self.plan),
                            "discard-moves": remainder.coordinates.shape[0]})
        for to_discard in remainder.coordinates:
            self.current_state = self.plan.add_discard(origin=to_discard, lattice=self.current_state)
        return self.plan, self.current_state
