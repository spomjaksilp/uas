"""

"""

import numpy as np
from numba import types, typed, typeof, njit
from uas.helper import type_path_matrix


@njit((types.ListType(type_path_matrix))(types.float32[:], types.float32[:]), fastmath=True)
def l_trajectory(origin, target):
    path = typed.List.empty_list(type_path_matrix)
    difference = origin - target
    # moves
    l_x = np.copy(difference)
    l_y = np.copy(difference)
    l_x[1] = 0
    l_y[0] = 0
    path.append(l_x)
    path.append(l_y)
    return path


@njit(types.float32(types.float32[:, :], types.float32[:]), fastmath=True)
def calculate_path_length(path, spacing):
    path = np.abs(path)
    return spacing[0] * np.sum(path[:, 0]) + spacing[1] * np.sum(path[:, 1])


class Trajectory:
    """
    Trajectories describe the way from a coordinate (x0, y0) to (x1, x1) via vectors
    """

    def __init__(self, origin: np.ndarray, target: np.ndarray, spacing: np.ndarray,
                 t_pick: float, t_place: float, v_move: float):
        self.origin = origin
        self.target = target
        self.spacing = spacing
        self.t_pick = t_pick
        self.t_place = t_place
        self.v_move = v_move
        self.path = []
        self.timeline = []
        self.calculate_trajectory()
        self.build_timeline()

    def calculate_trajectory(self):
        """
        Business end
        :return:
        """
        raise NotImplementedError

    def build_timeline(self):
        """

        :return:
        """
        # pick
        timer = 0
        timer += self.t_pick
        self.timeline.append(np.array((timer, 1, 0, 0)))
        # moves
        for ds in self.path:
            timer += calculate_path_length(np.array((ds, ), dtype=np.float32), self.spacing) / self.v_move
            self.timeline.append(np.array((timer, 1, ds[0], ds[1])))
        # place
        timer += self.t_place
        self.timeline.append(np.array((timer, 0, 0, 0)))


class Type1(Trajectory):
    """
    Type 1 move according to Léséleuc (2018).
    Describes a move in between sites
    """

    def calculate_trajectory(self):
        self.path = self._calculate_trajectory(self.origin.astype(np.float32), self.target.astype(np.float32))

    @staticmethod
    @njit((types.ListType(type_path_matrix))(types.float32[:], types.float32[:]), fastmath=True)
    def _calculate_trajectory(origin, target):
        path = typed.List.empty_list(type_path_matrix)
        difference = target - origin
        # handle edge case: moving only one site
        if np.linalg.norm(difference) == 1:
            # path.append(np.stack((origin, target)))
            path.append(difference)
        else:
            diagonal_step = difference / np.abs(difference) / 2
            # handle edge case: straight line, i.e. one component of difference == 0
            for i in range(2):
                if difference[i] == 0:
                    diagonal_step[i] = - 0.5
            # moves
            # diagonals move (step outside grid)
            post_origin = np.abs(origin + diagonal_step)
            pre_target = np.abs(target - diagonal_step)
            # L shaped path between the two intermediate points
            l_x, l_y = l_trajectory(post_origin, pre_target)
            path.append(diagonal_step)
            path.append(l_x)
            path.append(l_y)
            path.append(-1 * diagonal_step)
        return path


class Type2(Trajectory):
    """
    Type 2 move according to Léséleuc (2018).
    Describes a move on the connecting edges of sites
    """

    def calculate_trajectory(self):
        self.path = l_trajectory(self.origin.astype(np.float32), self.target.astype(np.float32))


class Discard(Type2):
    """
    Discard trajectory
    """
    def calculate_trajectory(self):
        self.path = [np.array((0.5, 0.5))]
