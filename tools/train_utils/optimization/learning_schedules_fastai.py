# This file is modified from https://github.com/traveller59/second.pytorch

import math
from functools import partial

import numpy as np
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper


class LRSchedulerStep(object):
    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):

        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):

            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):

            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0

    def step(self, step):
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)

class SegmentedOneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start, num_cycles, delay_matrix=None):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start/num_cycles
        self.num_cycles = num_cycles

        low_lr = self.lr_max / self.div_factor
        cycle_steps = total_step // num_cycles

        lr_phases = []
        mom_phases = []

        for i in range(num_cycles):
            a1 = int(cycle_steps * self.pct_start)
            a2 = cycle_steps - a1
            start_ = i / num_cycles

            start_step = i * cycle_steps
            end_step = (i + 1) * cycle_steps if i < num_cycles - 1 else total_step

            current_delay = delay_matrix[i]
            lr_max_ad = self.lr_max * current_delay
            low_lr_ad = low_lr * current_delay

            lr_phases.append((start_, partial(annealing_cos, low_lr_ad, lr_max_ad)))
            lr_phases.append((start_+self.pct_start, partial(annealing_cos, lr_max_ad, low_lr_ad / 1e4)))

            mom_phases.append((start_, partial(annealing_cos, *self.moms)))
            mom_phases.append((start_+self.pct_start, partial(annealing_cos, *self.moms[::-1])))

        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]

        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class FakeOptim:
    def __init__(self):
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()
    schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    plt.show()
    plt.plot(moms)
    plt.show()
