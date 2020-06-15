"""
This short example creates packed square target lattices of various sizes and simulates their scaling behaviour
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from uas import Lattice, Frontend
from uas.helper import Sample
from uas.strategy import French

DO_CALCULATION = False
p_loading = 0.5
n_runs = 100
shape_min = 5
shape_max = 10

shapes = np.arange(shape_min, shape_max + 1, 1)
means = []
stds = []

out = []

if DO_CALCULATION:
    for shape_target in shapes:
        print(shape_target)
        # calculate shape of sample from size roughly ceil of 1/sqrt(p) * size[0] to get about aus twice as many atoms
        shape_sample = math.ceil(1 / math.sqrt(p_loading) * shape_target)
        target_sample = Sample(shape=(shape_sample, shape_sample))
        target_sample.add_rect(origin=(2, 2), size=(shape_target, shape_target))
        target_state = Lattice(target_sample.value)
        completion_time = []
        for i in range(n_runs):
            start_sample = Sample(shape=(shape_sample, shape_sample))
            start_sample.add_random(probability=0.5)
            start_state = Lattice(start_sample.value)
            try:
                strat = French(start=deepcopy(start_state), target=target_state)
                plan, end_state = strat.run()
                frontend = Frontend(plan=plan, spacing=start_state.spacing)
                timeline = frontend.parse_plan()
                completion_time.append(timeline[-1][0])
            except AssertionError:
                # cannot sort, add NaN to list
                completion_time.append(np.NaN)
        out.append([shape_target, shape_sample, np.nanmean(completion_time), np.nanstd(completion_time),
                    *completion_time])

    np.save(file="packed_square_scaling_data.npy", arr=np.array(out))

data = np.load("packed_square_scaling_data.npy", allow_pickle=True)

print(data[:, 0:4])
# fig, ax = plt.subplots()
# ax.errorbar(shapes, (means / 1e3), yerr=(stds / 1e3))
#
# plt.tight_layout()
# plt.show()
#
