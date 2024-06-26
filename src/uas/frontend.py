"""

"""

from copy import deepcopy
import numpy as np
from uas import Plan, Type1, Discard, Lattice


class Frontend:
    """
    Accepts a plan, calculates the trajectories and timeline
    """
    def __init__(self, spacing: np.ndarray, plan: Plan):
        self.spacing = spacing
        self.plan = plan
        self.v_move = 100/1e3  # [µm/µs] 100µm/ms
        self.t_pick = 300  # [µs] 300µs
        self.t_place = 300  # [µs] 300µs
        self.timeline = np.ndarray(shape=(2, 2), dtype=np.float32)

    def parse_plan(self):
        timeline = []
        timer = 0
        for step in self.plan.moves:
            if step.discard:
                trajectory = Discard(origin=step.origin, target=step.target, spacing=self.spacing,
                                     t_start=timer, t_pick=self.t_pick, t_place=(self.t_place / 10),
                                     v_move=self.v_move)
            else:
                trajectory = Type1(origin=step.origin, target=step.target, spacing=self.spacing,
                                   t_start=timer, t_pick=self.t_pick, t_place=(self.t_place / 10),
                                   v_move=self.v_move)
            timeline.extend(trajectory.timeline)
            timer = timeline[-1][0]
        self.timeline = np.array(timeline, dtype=np.float32)
        return timeline


class Simulator(Frontend):
    """
    Accepts a plan, simulates the lattice
    """
    def __init__(self, lattice: Lattice, plan: Plan):
        super().__init__(plan=plan, spacing=lattice.spacing)
        self.lattice = lattice
        self.state = []

    def parse_plan(self):
        timeline = []
        timer = 0
        for step in self.plan.moves:
            if step.discard:
                self.lattice.discard(step.origin)
                trajectory = Discard(origin=step.origin, target=step.target, spacing=self.spacing,
                                     t_start=timer, t_pick=self.t_pick, t_place=(self.t_place / 10),
                                     v_move=self.v_move)
            else:
                self.lattice.move(step.origin, step.target)
                trajectory = Type1(origin=step.origin, target=step.target, spacing=self.spacing,
                                   t_start=timer, t_pick=self.t_pick, t_place=(self.t_place / 10),
                                   v_move=self.v_move)
            timeline.extend(trajectory.timeline)
            timer = timeline[-1][0]
            self.state.append({
                "time": timeline[-1][0],
                "lattice": deepcopy(self.lattice)
            })
        self.timeline = np.array(timeline, dtype=np.float32)
        return timeline
