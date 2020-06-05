"""

"""

import pytest
import numpy as np
from copy import deepcopy
from uas import Lattice, Simulator
from uas.helper import Sample
from uas.strategy import French


@pytest.fixture
def simulator():
    shape = (15, 15)
    size = (10, 10)
    start_state = Lattice(np.load(f"./data/sample_{shape[0]}_{shape[1]}.npy"))
    sample = Sample(shape)
    sample.add_rect(origin=(2, 2), size=size)
    target_state = Lattice(sample.value)
    strat = French(start=deepcopy(start_state), target=target_state)
    plan, end_state = strat.run()
    return Simulator(lattice=start_state, plan=plan)


class TestSimulator:
    def test_simulator(self, simulator):
        # assemble
        simulator.parse_plan()
        print(simulator.timeline)
