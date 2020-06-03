"""

"""

import numpy as np
from numba import types, typed, typeof, njit
from uas.helper import type_path_matrix


@njit((types.ListType(type_path_matrix))(types.float32[:], types.float32[:]), fastmath=True)
def l_trajectory(origin, target):
    path = typed.List.empty_list(type_path_matrix)
    # path.append(origin)
    difference = origin - target
    pointer = np.copy(origin)
    # moves
    pointer[0] = pointer[0] - difference[0]
    path.append(np.stack((origin, pointer)))
    path.append(np.stack((pointer, target)))
    return path


class Trajectory:
    """
    Trajectories describe the way from a coordinate (x0, y0) to (x1, x1) via vectors
    """
    def __init__(self, origin: np.ndarray, target: np.ndarray):
        self.origin = origin
        self.target = target
        self.path = []
        self.calculate_trajectory()

    def calculate_trajectory(self):
        """
        Business end
        :return:
        """
        raise NotImplementedError


class Type1(Trajectory):
    """
    Type 1 move according to Léséleuc (2018).
    Describes a move in between sites
    """
    def calculate_moves(self):
        self.path = self._calculate_moves(self.origin.astype(np.float32), self.target.astype(np.float32))

    @staticmethod
    @njit((types.ListType(type_path_matrix))(types.float32[:], types.float32[:]), fastmath=True)
    def _calculate_moves(origin, target):
        path = typed.List.empty_list(type_path_matrix)
        difference = target - origin
        # handle edge case: moving only one site
        if np.linalg.norm(difference) == 1:
            path.append(np.stack(origin, target))
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
            path.append(np.stack((origin, post_origin)))
            # L shaped path between the two intermediate points
            l_x, l_y = l_trajectory(post_origin, pre_target)
            path.append(l_x)
            path.append(l_y)
            path.append(np.stack((pre_target, target)))
        return path


class Type2(Trajectory):
    """
    Type 2 move according to Léséleuc (2018).
    Describes a move on the connecting edges of sites
    """
    def calculate_moves(self):
        self.path = l_trajectory(self.origin.astype(np.float32), self.target.astype(np.float32))