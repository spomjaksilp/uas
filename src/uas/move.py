"""

"""

import numpy as np


class Move:
    """
    Defines a single move from a origin coordinate to a target coordinate
    """
    def __init__(self, origin: np.ndarray, target: np.ndarray, discard: bool=False):
        self.origin = origin
        self.target = target
        self.discard = discard


class Plan:
    """
    An ordered collection of moves
    """

    def __init__(self):
        self.moves = []

    def __len__(self):
        return len(self.moves)

    def add_move(self, origin: np.ndarray, target: np.ndarray):
        """
        Adds moves to the list of moves
        :param origin: (ndarray) coordinate
        :param target: (ndarray) coordinate
        :return: (Lattice)
        """
        self.moves.append(Move(origin=origin, target=target))

    def add_discard(self, origin: np.ndarray):
        """
        Adds a discard move to the list of moves
        :param origin: (ndarray) coordinate
        :return: (Lattice)
        """
        self.moves.append(Move(origin=origin, target=origin, discard=True))
