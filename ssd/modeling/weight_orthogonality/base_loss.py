# -*- coding:utf-8 -*-

""" base_loss """

from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC, abstractmethod

import torch
from torch.nn.modules.loss import _WeightedLoss


class BaseModelLoss(_WeightedLoss, ABC):
    def __init__(self, model, weight=None, size_average=None, reduce=None, reduction='mean'):
        if isinstance(weight, float):
            weight = torch.ones((), dtype=torch.float32) * weight
        self.kernels = self.get_kernels(model)
        super().__init__(weight, size_average, reduce, reduction)

    @abstractmethod
    def get_kernels(self, model):
        pass
