"""

"""

from uas import Lattice


class StrategyTemplate:
    """
    Template for all sorting strategies
    """

    def __init__(self, start: Lattice, target: Lattice):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
