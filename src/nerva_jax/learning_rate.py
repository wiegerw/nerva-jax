# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Learning-rate schedulers.

These schedulers are intentionally minimal and stateless unless noted
(TimeBased updates its internal lr). The parser accepts textual forms such as
"Constant(0.1)", "StepBased(0.1,0.5,10)", or "MultiStepLR(0.1;1,3,5;0.1)".
"""

import math
import re
from typing import List


class LearningRateScheduler(object):
    """Interface for epoch-indexed learning-rate schedules."""
    def __call__(self, epoch: int) -> float:
        raise NotImplementedError


class ConstantScheduler(LearningRateScheduler):
    """Constant learning rate: returns the same lr for any epoch."""
    def __init__(self, lr: float):
        self.lr = lr

    def __str__(self):
        return f'ConstantScheduler(lr={self.lr})'

    def __call__(self, epoch: int) -> float:
        return self.lr


class TimeBasedScheduler(LearningRateScheduler):
    """Time-based decay: lr = lr / (1 + decay * epoch)."""
    def __init__(self, lr: float, decay: float):
        self.lr = lr
        self.decay = decay

    def __str__(self):
        return f'TimeBasedScheduler(lr={self.lr}, decay={self.decay})'

    def __call__(self, epoch: int) -> float:
        self.lr = self.lr / (1 + self.decay * float(epoch))
        return self.lr


class StepBasedScheduler(LearningRateScheduler):
    """Step decay: lr * drop_rate ^ floor((1+epoch)/change_rate)."""
    def __init__(self, lr: float, drop_rate: float, change_rate: float):
        self.lr = lr
        self.drop_rate = drop_rate
        self.change_rate = change_rate

    def __str__(self):
        return f'StepBasedScheduler(lr={self.lr}, drop_rate={self.drop_rate}, change_rate={self.change_rate})'

    def __call__(self, epoch: int) -> float:
        return self.lr * math.pow(self.drop_rate, math.floor((1.0 + epoch) / self.change_rate))


class MultiStepLRScheduler(LearningRateScheduler):
    """Multi-step decay: multiply lr by gamma at specified milestone epochs."""
    def __init__(self, lr: float, milestones: List[int], gamma: float):
        self.lr = lr
        self.milestones = milestones
        self.gamma = gamma

    def __str__(self):
        return f'MultiStepLRScheduler(lr={self.lr}, milestones={self.milestones}, gamma={self.gamma})'

    def __call__(self, epoch: int) -> float:
        eta = self.lr
        for milestone in self.milestones:
            if epoch >= milestone:
                eta *= self.gamma
            else:
                break
        return eta


class ExponentialScheduler(LearningRateScheduler):
    """Exponential decay: lr * exp(-change_rate * epoch)."""
    def __init__(self, lr: float, change_rate: float):
        self.lr = lr
        self.change_rate = change_rate

    def __str__(self):
        return f'ExponentialScheduler(lr={self.lr}, change_rate={self.change_rate})'

    def __call__(self, epoch: int) -> float:
        return self.lr * math.exp(-self.change_rate * float(epoch))


def parse_learning_rate(text: str) -> LearningRateScheduler:
    """Parse a textual learning-rate scheduler specification.

    Accepted forms include Constant(lr), TimeBased(lr,decay),
    StepBased(lr,drop_rate,change_rate), MultiStepLR(lr;milestones;gamma)
    and Exponential(lr,change_rate).
    """
    try:
        if text.startswith('Constant'):
            m = re.match(r'Constant\((.*)\)', text)
            lr = float(m.group(1))
            return ConstantScheduler(lr)
        elif text.startswith('TimeBased'):
            m = re.match(r'TimeBased\((.*),(.*)\)', text)
            lr = float(m.group(1))
            decay = float(m.group(2))
            return TimeBasedScheduler(lr, decay)
        elif text.startswith('StepBased'):
            m = re.match(r'StepBased\((.*),(.*),(.*)\)', text)
            lr = float(m.group(1))
            drop_rate = float(m.group(2))
            change_rate = float(m.group(3))
            return StepBasedScheduler(lr, drop_rate, change_rate)
        elif text.startswith('MultiStepLR'):
            m = re.match(r'MultiStepLR\((.*);(.*);(.*)\)', text)
            lr = float(m.group(1))
            milestones = [int(x) for x in m.group(2).split(',')]
            gamma = float(m.group(3))
            return MultiStepLRScheduler(lr, milestones, gamma)
        elif text.startswith('Exponential'):
            m = re.match(r'Exponential\((.*),(.*)\)', text)
            lr = float(m.group(1))
            change_rate = float(m.group(2))
            return ExponentialScheduler(lr, change_rate)
    except:
        pass
    raise RuntimeError(f"could not parse learning rate scheduler '{text}'")
