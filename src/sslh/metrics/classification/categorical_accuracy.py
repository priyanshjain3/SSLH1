#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch

from torch import Tensor

from sslh.metrics.base import Metric


class CategoricalAccuracy(Metric):
    def __init__(
        self,
        dim: int = -1,
        vector_input: bool = True,
        vector_target: bool = False,
        reduce_fn: Optional[Callable] = torch.mean,
    ) -> None:
        """
        Compute the categorical accuracy between a batch of prediction and labels.

        :param dim: The dimension to compute the score. (default: -1)
        :param vector_input: If True, considers inputs as a vector of probabilities.
                If False, it will be considered as a vectors of classes index. (default: True)
        :param vector_target: If True, considers target as a vector of probabilities.
                If False, it will be considered as a vectors of classes index. (default: True)
        :param reduce_fn: The reduction function to apply. (default: torch.mean)
        """
        super().__init__()
        self.dim = dim
        self.vector_input = vector_input
        self.vector_target = vector_target
        self.reduce_fn = reduce_fn

    def compute_score(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.vector_input:
            pred = pred.argmax(dim=self.dim)
        if self.vector_target:
            target = target.argmax(dim=self.dim)
        elif len(pred.shape) == 1 and len(target.shape) > 1:
            target = target.argmax(dim=1)
        elif len(target.shape) == 1 and len(pred.shape) > 1:
            target_onehot = torch.zeros(target.size(0), pred.size(1), device=target.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)
            target = target_onehot
        assert pred.shape == target.shape, f"Input and target must have the same shape. Got pred.shape={pred.shape} and target.shape={target.shape}."
        assert 0 <= len(pred.shape) <= 2

        score = pred.eq(target).float()
        if self.reduce_fn is not None:
            score = self.reduce_fn(score)
        return score
