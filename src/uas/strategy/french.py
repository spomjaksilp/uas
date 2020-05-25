"""
From Paris
"""

import numpy as np
from scipy.spatial import distance
from numba import njit, types, typeof, typed
from uas import Lattice, Plan, Type2
from uas.helper import type_coordinate, type_coordinate_matrix, type_site_matrix
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

    @staticmethod
    def calculate_distances(start_coordinates, target_coordinates):
        """
        Calculates distances between start sites and target sites
        :param start_coordinates:
        :param target_coordinates:
        :return:
        """
        dist = distance.cdist(start_coordinates, target_coordinates, 'cityblock').astype(np.int16)
        # dist_sorted = np.argsort(dist.flatten())  # use in case this will be jit-ed
        dist_sorted = np.argsort(dist, axis=None)
        # https://stackoverflow.com/questions/29734660/python-numpy-keep-a-list-of-indices-of-a-sorted-2d-array
        coordinates_sorted = np.vstack(np.unravel_index(dist_sorted, dist.shape)).T
        out = coordinates_sorted.astype(np.int16)
        return out

    @staticmethod
    @njit((types.ListType(type_coordinate))(type_coordinate_matrix, type_site_matrix, type_site_matrix,
                                            type_coordinate_matrix, type_coordinate_matrix))
    def loop_over_distances(distances, start_visited, target_visited, start_coordinates, target_coordinates):
        """
        Loop over sorted distances between start and target, build list
        This loop is offloaded into a static function to make it jit-compatible
        :param distances:
        :param start_visited:
        :param target_visited:
        :param start_coordinates:
        :param target_coordinates:
        :return:
        """
        queue = typed.List.empty_list(type_coordinate)
        for coord in distances:
            s_0, s_1 = start_coordinates[coord[0]], target_coordinates[coord[1]]
            if start_visited[s_0[0], s_0[1]] or target_visited[s_1[0], s_1[1]]:
                # already sorted or there is no atom, go on
                continue
            start_visited[s_0[0], s_0[1]] = 1
            target_visited[s_1[0], s_1[1]] = 1
            queue.append(np.append(s_0, s_1))
        return queue

    def run(self):
        # mark already sorted sites
        start_visited = np.zeros_like(self.start.value, dtype=np.bool)
        start_visited[np.logical_and(self.start.value, self.target.value)] = 1
        target_visited = np.copy(start_visited)
        distances = self.calculate_distances(self.start.coordinates, self.target.coordinates)
        queue = self.loop_over_distances(distances, start_visited, target_visited,
                                         self.start.coordinates, self.target.coordinates)
        for coord in queue:
            s_0 = coord[:2]
            s_1 = coord[2:]
            # add a move to the plan
            self.plan.add_move(origin=s_0, target=s_1, lattice=self.current_state)
            self.current_state.move(origin=s_0, target=s_1)
        # drop left over atoms
        remainder = Lattice(np.logical_and(self.current_state.value, np.logical_not(self.target.value)),
                            spacing=self.current_state.spacing)
        self.report.update({"site-moves": len(self.plan),
                            "discard-moves": remainder.coordinates.shape[0]})
        for to_discard in remainder.coordinates:
            self.plan.add_discard(origin=to_discard, lattice=self.current_state)
            self.current_state.discard(origin=to_discard)
        return self.plan, self.current_state
