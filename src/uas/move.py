"""

"""

import numpy as np
from numba import types, typed, typeof
from numba.experimental import jitclass
from uas import Lattice

type_coordinate = types.int16[:]

spec_move = [
    ("origin", types.int16[:]),
    ("target", types.int16[:]),
    ("path", types.ListType(type_coordinate)),
]


class Move:
    """
    Defines a single move from a origin coordinate to a target coordinate
    """

    def __init__(self, origin: np.ndarray, target: np.ndarray):
        self.origin = origin
        self.target = target
        self.path = typed.List.empty_list(type_coordinate)
        self.calculate_moves()

    def calculate_moves(self):
        """
        Calculates the internal trajectory
        :return:
        """
        raise NotImplementedError


@jitclass(spec_move)
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
        pass


@jitclass(spec_move)
class Type2(Move):
    """
    Type 2 move according to Léséleuc (2018).
    Describes a move on the connecting edges of sites
    """

    def calculate_moves(self):
        self.path.append(self.origin)
        difference = self.origin - self.target
        pointer = np.zeros_like(self.origin) + self.origin
        for component in difference:
            # move
            pointer += component
            self.path.append(pointer)


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
        return Lattice.move(lattice, origin, target)

    def add_discard(self, origin: np.ndarray, lattice: Lattice):
        """
        Adds a discard move to the list of moves
        :param origin: (ndarray) coordinate
        :param lattice: (Lattice)
        :return: (Lattice)
        """
        self._moves.append(self.discard(origin=origin, target=origin))
        return Lattice.move(lattice, origin, target=None)
