"""

"""

from copy import deepcopy
from uas import Lattice, Plan, Move, Type1, Discard


class Simulator:
    """
    Accepts a plan, calculates the trajectories and duration
    """
    def __init__(self, lattice: Lattice, plan: Plan):
        self.lattice = lattice
        self.plan = plan
        self.v_move = 100e-6/1e-3  # 100µm/ms
        self.t_pick = 300e-6  # 300µs
        self.t_place = 300e-6  # 300µs
        self.state = []
        self.timeline = []

    def parse_plan(self):
        for step in self.plan.moves:
            if step.discard:
                self.lattice.discard(step.origin)
                trajectory = Discard(origin=step.origin, target=step.target, lattice=self.lattice,
                                     t_pick=self.t_pick, t_place=(self.t_place / 10), v_move=self.v_move)
            else:
                self.lattice.move(step.origin, step.target)
                trajectory = Type1(origin=step.origin, target=step.target, lattice=self.lattice,
                                   t_pick=self.t_pick, t_place=self.t_place, v_move=self.v_move)
            self.timeline.extend(trajectory.timeline)
            self.state.append({
                "time": self.timeline[-1][0],
                "lattice": deepcopy(self.lattice)
            })
            print(f"timer {self.timeline[-1][0] * 1e3}ms")

