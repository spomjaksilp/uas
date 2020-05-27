"""

"""

import numpy as np
from numba import types, typed, typeof, njit
from numba.experimental import jitclass
from uas import Lattice
from uas.helper import type_coordinate, type_coordinate_matrix

spec_move = [
    ("origin", types.int16[:]),
    ("target", types.int16[:]),
    ("path", types.ListType(type_coordinate)),
]


@njit((types.ListType(type_coordinate_matrix))(type_coordinate, type_coordinate), fastmath=True)
def l_path(origin, target):
    path = typed.List.empty_list(type_coordinate_matrix)
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
        self.path.append(self.origin)


class Type1(Move):
    """
    Type 1 move according to Léséleuc (2018).
    Describes a move in between sites
    """
    def calculate_moves(self):
        self.path = self._calculate_moves(self.origin, self.target)

    @staticmethod
    @njit((types.ListType(type_coordinate_matrix))(type_coordinate, type_coordinate), fastmath=True)
    def _calculate_moves(origin, target):
        path = typed.List.empty_list(type_coordinate_matrix)
        # path.append(origin)
        difference = target - origin
        diagonal_step = difference / np.abs(difference)
        # moves
        # diagonals move (step outside grid)
        post_origin = origin + diagonal_step
        pre_target = target - diagonal_step
        path.append(np.stack((origin, post_origin.astype(np.int16))))
        # L shaped path between the two intermediate points
        for piece in l_path(post_origin.astype(np.int16), pre_target.astype(np.int16)):
            path.append(piece)
        # path = path + l_path(post_origin.astype(np.int16), pre_target.astype(np.int16))
        path.append(np.stack((pre_target.astype(np.int16), target)))
        return path


# @jitclass(spec_move)
class Type2(Move):
    """
    Type 2 move according to Léséleuc (2018).
    Describes a move on the connecting edges of sites
    """
    def calculate_moves(self):
        self.path = l_path(self.origin, self.target)


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
