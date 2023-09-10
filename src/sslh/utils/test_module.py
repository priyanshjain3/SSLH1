#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from pytorch_lightning import LightningModule
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple

from sslh.nn.utils import ForwardDictAffix
from sslh.utils.custom_logger import CustomTensorboardLogger


class TestModule(LightningModule):
    def __init__(
        self,
        module: nn.Module,
        metric_dict: Optional[Dict[str, nn.Module]],
        prefix: str,
    ) -> None:
        """
        LightningModule wrapper for module and a metric dict.

        Example :

        >>> from pytorch_lightning import Trainer
        >>> from sslh.metrics.classification.categorical_accuracy import CategoricalAccuracy
        >>> model = ...
        >>> trainer = Trainer(...)
        >>> test_dataloader = ...
        >>> test_module = TestModule(model, ForwardDictAffix(acc=CategoricalAccuracy()))
        >>> trainer.test(test_module, test_dataloader)

        :param module: The module to wrap for testing the forward output.
        :param metric_dict: The metric dict object.
        :param prefix: The prefix used in metrics names.
        """
        super().__init__()
        self.module = module
        self.metric_dict = ForwardDictAffix(metric_dict, prefix=prefix)

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        xs, ys = batch
        probs_xs = self(xs)
        scores = self.metric_dict(probs_xs, ys)
        self.log_dict(
            scores, on_epoch=True, on_step=False, logger=False, prog_bar=False
        )
        return scores

    def test_epoch_end(self, scores_lst: List[Dict[str, Tensor]]) -> None:
        scores = {
            name: torch.stack([scores[name] for scores in scores_lst]).mean().item()
            for name in scores_lst[0].keys()
        }
        if isinstance(self.logger, CustomTensorboardLogger):
            self.logger.log_hyperparams({}, scores)
        else:
            self.logger.log_metrics(scores, None)  # type: ignore

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
