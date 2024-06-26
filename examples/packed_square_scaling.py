"""
This short example creates packed square target lattices of various sizes and simulates their scaling behaviour
"""

from functools import partial
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import deepcopy
from uas import Lattice, Frontend
from uas.helper import Sample
from uas.strategy import French
from multiprocessing import cpu_count
from joblib import Parallel, delayed


def run(target_state: Sample, p_loading):
    """
    Execute one sorting into target_state with given loading probability and return the required time
    """
    start_sample = Sample(shape=target_state.value.shape)
    start_sample.add_random(probability=0.5)
    start_state = Lattice(start_sample.value)
    try:
        strat = French(start=deepcopy(start_state), target=target_state)
        plan, end_state = strat.run()
        frontend = Frontend(plan=plan, spacing=start_state.spacing)
        timeline = frontend.parse_plan()
        completion_time = timeline[-1][0]
    except AssertionError:
        # cannot sort, add NaN to list
        completion_time = np.NaN
    return completion_time


DO_CALCULATION = True
DO_PLOTTING = True
p_loading = 0.5
# runs per shape 
n_runs = 20
# threads
n_jobs = cpu_count() - 1

# array shapes
shapes = np.around(np.logspace(start=1, stop=6, num=10, base=2), decimals=0)
out = []

if DO_CALCULATION:
    for shape_target in shapes:
        print(shape_target)
        # calculate shape of sample from size roughly ceil of 1/sqrt(p) * size[0] to get about aus twice as many atoms
        shape_sample = math.ceil(1 / math.sqrt(p_loading) * shape_target)
        target_sample = Sample(shape=(shape_sample, shape_sample))
        target_sample.add_rect(origin=(2, 2), size=(shape_target, shape_target))
        target_state = Lattice(target_sample.value)
        this_run = partial(run, target_state=target_state, p_loading = p_loading)
        completion_time = Parallel(n_jobs=n_jobs, backend="multiprocessing")(delayed(this_run)() for i in range(n_runs))
        out.append([shape_target, shape_sample, np.nanmean(completion_time), np.nanstd(completion_time),
                    *completion_time])

    np.save(file="packed_square_scaling_data.npy", arr=np.array(out))

if DO_PLOTTING:
    data = np.load("packed_square_scaling_data.npy", allow_pickle=True)

    sites = data[:, 0] ** 2
    means = data[:, 2] / 1e3
    stds = data[:, 3] / 1e3


    def scaling_func(x, a, b):
        return a * x ** b


    fig, ax = plt.subplots()
    ax.errorbar(sites, means, yerr=stds, fmt="o")
    p0 = [.1, .5]
    popt, perr = curve_fit(scaling_func, sites, means, p0)
    a, b = f"{popt[0]:.2f}", f"{popt[1]:.2f}"
    ax.plot(sites, scaling_func(sites, *popt), label="fitted scaling with ${%s}N^{%s}$" % (a, b))

    ax.set_title("Simulated assembly time into tight-packed square lattice\n"
                 "$v_{move}=100\\,\\mu s/ms$, $\\tau_{pick}=\\tau_{place}=300\\,\\mu s$")
    ax.set_xlabel("number N of target sites")
    ax.set_ylabel("assembly time [ms]")
    ax.legend(loc="lower right")
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig("example_scaling_behavior.png", dpi=300)

