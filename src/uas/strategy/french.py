"""
From Paris
"""

import numpy as np
from scipy.spatial import distance
from numba import njit, types, typeof
from uas import Lattice, Plan, Type2
from . import StrategyTemplate


# @njit((types.int16[:, :])(types.int16[:, :], types.int16[:]))
def get_coordinates_from_distances(coordinates, coordinate_pair):
    s_0 = coordinates[0][coordinate_pair[0]]
    s_1 = coordinates[1][coordinate_pair[1]]
    return np.array((s_0, s_1))

type_coordinate = types.int16[:]

@njit((types.ListType(type_coordinate))(types.int16[:, :], types.boolean[:, :], types.boolean[:, :, :], types.int16[:, :], types.int16[:, :]))
def loop_over_distances(distances, start_visited, target_visited, start_coordinates, target_coordinates):
    for coord in distances:
        s_0, s_1 = start_coordinates[coord[0]], target_coordinates[coord[1]]
        if start_visited[tuple(s_0)] or target_visited[tuple(s_1)]:
            # already sorted or there is no atom, go on
            continue
        start_visited[tuple(s_0)] = 1
        target_visited[tuple(s_1)] = 1
    out = typed.List.empty_list(type_coordinate)
    out.append(start_visited)
    out.append(target_visited)
    return out


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
        return coordinates_sorted.astype(np.int16)

    def run(self):
        # mark already sorted sites
        start_visited = np.where(self.start.value & self.target.value, np.ones_like(self.start.value),
                                  np.zeros_like(np.ones_like(self.start.value)))
        target_visited = np.copy(start_visited)
        # loop over sorted distances between start and target
        # _, __ = loop_over_distances(self.calculate_distances(), start_visited, target_visited,
        #                             self.start.coordinates, self.target.coordinates)
        for coord in self.calculate_distances():
            s_0, s_1 = self.start.coordinates[coord[0]], self.target.coordinates[coord[1]]
            if start_visited[tuple(s_0)] or target_visited[tuple(s_1)]:
                # already sorted or there is no atom, go on
                continue
            # add a move to the plan
            self.current_state = self.plan.add_move(origin=s_0, target=s_1, lattice=self.current_state)
            # mark sites as visited
            start_visited[tuple(s_0)] = 1
            target_visited[tuple(s_1)] = 1
        # drop left over atoms
        remainder = Lattice(np.logical_and(self.current_state.value, np.logical_not(self.target.value)),
                            spacing=self.current_state.spacing)
        self.report.update({"site-moves": len(self.plan),
                            "discard-moves": remainder.coordinates.shape[0]})
        for to_discard in remainder.coordinates:
            self.current_state = self.plan.add_discard(origin=to_discard, lattice=self.current_state)
        return self.plan, self.current_state
