import torch
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateScheduler(_LRScheduler):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

class WarmupStepLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.

    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
            decay_steps: int,
            decay_scale: float=0.1
    ) -> None:
        super(WarmupStepLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = init_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_scale = decay_scale

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        if self.update_steps > self.warmup_steps and self.update_steps % self.decay_steps == 0:
            lr = self.lr * self.decay_scale
            self.set_lr(self.optimizer, lr)
            self.lr = lr
            
        self.update_steps += 1
        return self.lr