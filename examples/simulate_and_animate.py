"""

"""

import numpy as np
from copy import deepcopy
from uas import Lattice, Simulator
from uas.helper import Sample, animate_timeline
from uas.strategy import French


shape = (20, 20)
size = (10, 10)
start_state = Lattice(np.load(f"./data/sample_{shape[0]}_{shape[1]}.npy"))
sample = Sample(shape)
sample.add_rect(origin=(2, 2), size=size)
target_state = Lattice(sample.value)
strat = French(start=deepcopy(start_state), target=target_state)
plan, end_state = strat.run()
simulator = Simulator(lattice=start_state, plan=plan)

simulator.parse_plan()
animate_timeline(simulator.state)
