"""

"""

import numpy as np
from uas import Lattice


class Move:
    """
    Defines a single move from a origin coordinate to a target coordinate
    """

    def __init__(self, origin: np.ndarray, target: np.ndarray or None = None):
        self.origin = origin
        self.target = origin if target is None else target
        self._path = []
        self.calculate_moves()

    def calculate_moves(self):
        """
        Calculates the internal trajectory
        :return:
        """
        raise NotImplementedError


class Discard(Move):
    """
    Discard an atom in a site
    """
    def calculate_moves(self):
        self._path.append(self.origin)


class Type1(Move):
    """
    Type 1 move according to Léséleuc (2018).
    Describes a move in between sites
    """
    def calculate_moves(self):
        pass


class Type2(Move):
    """
    Type 2 move according to Léséleuc (2018).
    Describes a move on the connecting edges of sites
    """
    def calculate_moves(self):
        self._path.append(self.origin)
        difference = self.origin - self.target
        pointer = np.zeros_like(self.origin) + self.origin
        for component in difference:
            # move
            pointer += component
            self._path.append(pointer)


class Plan:
    """
    An ordered collection of moves
    """

    def __init__(self, move: Move = Move, discard = Discard):
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
        self._moves.append(self.discard(origin=origin))
        return Lattice.move(lattice, origin, target=None)
