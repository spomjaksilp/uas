"""

"""

import numpy as np


class Move:
    """
    Defines a single move from a origin coordinate to a target coordinate
    """
    def __init__(self, origin: np.ndarray, target: np.ndarray):
        self.origin = origin
        self.target = target


class Discard(Move):
    """
    Discard an atom in a site
    """
    pass


class Plan:
    """
    An ordered collection of moves
    """

    def __init__(self):
        self._moves = []

    def __len__(self):
        return len(self._moves)

    def add_move(self, origin: np.ndarray, target: np.ndarray):
        """
        Adds moves to the list of moves
        :param origin: (ndarray) coordinate
        :param target: (ndarray) coordinate
        :return:
        """
        self._moves.append(self.move(origin=origin, target=target))

    def add_discard(self, origin: np.ndarray):
        """
        Adds a discard move to the list of moves
        :param origin: (ndarray) coordinate
        :return:
        """
        self._moves.append(self.discard(origin=origin, target=origin))
