"""
Fairness accounting for LOSO experiments (reviewer request 2).

Every condition logs what it consumed during training so that
zero-shot comparisons can be made compute-matched and
data-exposure-matched, not just final-accuracy-matched.

Counters:
    backbone_steps  — AdamW steps over the full model
    inner_steps     — per-adapter SGD steps (Reptile inner loop)
    reptile_updates — number of Reptile centroid updates
    trials_seen     — cumulative labelled trials consumed
    subjects_seen   — distinct training subjects seen
    wall_clock_s    — seconds of training time

The returned dict is wandb-loggable directly.
"""

import time


class ExposureTracker:
    def __init__(self):
        self.backbone_steps  = 0
        self.inner_steps     = 0
        self.reptile_updates = 0
        self.trials_seen     = 0
        self._subjects_seen  = set()
        self._t0             = None

    def start(self):
        self._t0 = time.perf_counter()

    def record_backbone_step(self, batch_size, sids):
        self.backbone_steps += 1
        self.trials_seen    += int(batch_size)
        self._subjects_seen.update(int(s) for s in sids)

    def record_inner_step(self, batch_size, sid):
        self.inner_steps += 1
        self.trials_seen += int(batch_size)
        self._subjects_seen.add(int(sid))

    def record_reptile_update(self):
        self.reptile_updates += 1

    def summary(self):
        wall = (time.perf_counter() - self._t0) if self._t0 else 0.0
        return {
            'exposure/backbone_steps':  self.backbone_steps,
            'exposure/inner_steps':     self.inner_steps,
            'exposure/total_grad_steps': self.backbone_steps + self.inner_steps,
            'exposure/reptile_updates': self.reptile_updates,
            'exposure/trials_seen':     self.trials_seen,
            'exposure/subjects_seen':   len(self._subjects_seen),
            'exposure/wall_clock_s':    wall,
        }
