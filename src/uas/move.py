"""

"""

import numpy as np
from uas import Lattice


class Move:
    """
    Defines a single move from a origin coordinate to a target coordinate
    """

    def __init__(self, origin: list, target: list):
        self.origin = origin
        self.target = target
        self._path = []
        self.calculate_moves()

    def calculate_moves(self):
        """
        Calculates the internal trajectory
        :return:
        """
        raise NotImplementedError


class Plan:
    """
    An ordered collection of moves
    """

    def __init__(self, move: Move = Move):
        self.move = move
        self._moves = []

    def __len__(self):
        return len(self._moves)

    def add_move(self, origin: list, target: list, lattice: Lattice):
        """
        Add moves to internal path
        :param origin: (list) coordinate
        :param target: (list) coordinate
        :param lattice:
        :return:
        """
        self._moves.append(self.move(origin=origin, target=target))
        return Lattice.move(lattice, origin, target)

