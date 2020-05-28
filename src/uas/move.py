"""

"""

import numpy as np
from numba import types, typed, typeof, njit
from numba.experimental import jitclass
from uas import Lattice
from uas.helper import type_coordinate, type_path_matrix

spec_move = [
    ("origin", types.int16[:]),
    ("target", types.int16[:]),
    ("path", types.ListType(type_coordinate)),
]


@njit((types.ListType(type_path_matrix))(types.float32[:], types.float32[:]), fastmath=True)
def l_path(origin, target):
    path = typed.List.empty_list(type_path_matrix)
    # path.append(origin)
    difference = origin - target
    pointer = np.copy(origin)
    # moves
    pointer[0] = pointer[0] - difference[0]
    path.append(np.stack((origin, pointer)))
    path.append(np.stack((pointer, target)))
    return path


class Move:
    """
    Defines a single move from a origin coordinate to a target coordinate
    """

    def __init__(self, origin: np.ndarray, target: np.ndarray):
        self.origin = origin
        self.target = target
        self.path = []
        # self.path = typed.List.empty_list(type_coordinate)  # numba
        self.calculate_moves()

    def calculate_moves(self):
        """
        Calculates the internal trajectory
        :return:
        """
        raise NotImplementedError


# @jitclass(spec_move)
class Discard(Move):
    """
    Discard an atom in a site
    """

    def calculate_moves(self):
        self.path.append(np.stack((self.origin, self.origin)))


class Type1(Move):
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
        # path.append(origin)
        difference = target - origin
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
        l_x, l_y = l_path(post_origin, pre_target)
        path.append(l_x)
        path.append(l_y)
        path.append(np.stack((pre_target, target)))
        return path


# @jitclass(spec_move)
class Type2(Move):
    """
    Type 2 move according to Léséleuc (2018).
    Describes a move on the connecting edges of sites
    """
    def calculate_moves(self):
        self.path = l_path(self.origin.astype(np.float32), self.target.astype(np.float32))


class Plan:
    """
    An ordered collection of moves
    """

    def __init__(self, move: Move = Move, discard=Discard):
        self.move = move
        self.discard = discard
        self._moves = []

    def __len__(self):
        return len(self._moves)

    def add_move(self, origin: np.ndarray, target: np.ndarray, lattice: Lattice):
        """
        Adds moves to the list of moves
        :param origin: (ndarray) coordinate
        :param target: (ndarray) coordinate
        :param lattice: (Lattice)
        :return: (Lattice)
        """
        self._moves.append(self.move(origin=origin, target=target))

    def add_discard(self, origin: np.ndarray, lattice: Lattice):
        """
        Adds a discard move to the list of moves
        :param origin: (ndarray) coordinate
        :param lattice: (Lattice)
        :return: (Lattice)
        """
        self._moves.append(self.discard(origin=origin, target=origin))
