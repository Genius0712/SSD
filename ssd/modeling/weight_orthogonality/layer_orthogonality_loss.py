# -*- coding:utf-8 -*-

""" layer_orthogonality_losses """

from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC
import numpy as np
import torch
from torch import nn

from .base_loss import BaseModelLoss


class _LayerOrthogonalityLoss(BaseModelLoss, ABC):
    def get_kernels(self, model):
        return [layer.weight for layer in model.modules() if isinstance(layer, nn.Conv2d)]


class LayerOrthogonalityPullLoss(_LayerOrthogonalityLoss):
    def forward(self):
        total_loss = []
        for kernel in self.kernels:
            kernel_mat = kernel.flatten(start_dim=1)  # Shape:  Cout x (Cin*k*k)
            kernel_mat = kernel_mat / torch.norm(kernel_mat)  # Cout x (Cin*k*k)
            kernel_mat_correlation = torch.mm(kernel_mat, torch.transpose(kernel_mat, 1, 0))  # shape: Cout × Cout
            kernel_mat_correlation = torch.triu(kernel_mat_correlation, diagonal=1)  # 实对称阵下三角置0 Cout × Cout
            total_loss.append(
                torch.norm(kernel_mat_correlation)  # scalar
            )
        total_loss = torch.stack(total_loss)
        if self.reduction != 'none':
            total_loss = (torch.mean(total_loss)
                          if self.reduction == 'mean' else torch.sum(total_loss))
        return total_loss * self.weight


class LayerOrthogonalityPushLoss(_LayerOrthogonalityLoss):
    pass
