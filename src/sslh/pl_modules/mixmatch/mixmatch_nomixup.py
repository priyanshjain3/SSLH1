#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import torch

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.pl_modules.mixmatch.mixmatch import MixMatch


class MixMatchNoMixup(MixMatch):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        activation: nn.Module = nn.Softmax(dim=-1),
        criterion_s: nn.Module = CrossEntropyLossVecTargets(reduction="mean"),
        criterion_u: nn.Module = CrossEntropyLossVecTargets(reduction="mean"),
        lambda_u: float = 1.0,
        n_augms: int = 2,
        temperature: float = 0.5,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        MixMatch without Mixup (MMN) LightningModule.

        :param model: The PyTorch Module to train.
                The forward() must return logits to classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: Softmax(dim=-1))
        :param criterion_s: The loss component 'L_s' of MM.
                (default: CrossEntropyWithVectors())
        :param criterion_u: The loss component 'L_u' of MM.
                (default: CrossEntropyWithVectors())
        :param lambda_u: The coefficient of the 'L_u' component. (default: 1.0)
        :param n_augms: The number of strong augmentations applied. (default: 2)
        :param temperature: The temperature applied by the sharpen function.
                A lower temperature make the pseudo-label produced more 'one-hot'.
                (default: 0.5)
        :param train_metrics: An optional dictionary of metrics modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics modules for validation.
                (default: None)
        :param log_on_epoch: If True, log only the epoch means of each train metric score.
                (default: True)
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            activation=activation,
            criterion_s=criterion_s,
            criterion_u=criterion_u,
            lambda_u=lambda_u,
            n_augms=n_augms,
            temperature=temperature,
            alpha=0.0,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            log_on_epoch=log_on_epoch,
        )

    def training_step(
        self, batch: Tuple[Tuple[Tensor, Tensor], List[Tensor]], batch_idx: int
    ) -> Tensor:
        (xs_weak, ys), xu_weak_lst = batch

        with torch.no_grad():
            # Guess pseudo-label 'yu'
            yu = self.guess_label(xu_weak_lst)
            yu_lst = yu.repeat([self.n_augms] + [1] * (len(yu.shape) - 1))

            # Stack augmented 'xu' variants to a single batch
            xu_weak_lst = torch.vstack(xu_weak_lst)

        logits_xs_weak = self.model(xs_weak)
        logits_xu_weak_lst = self.model(xu_weak_lst)

        probs_xs_weak = self.activation(logits_xs_weak)
        probs_xu_weak_lst = self.activation(logits_xu_weak_lst)

        loss_s = self.criterion_s(probs_xs_weak, ys)
        loss_u = self.criterion_u(probs_xu_weak_lst, yu_lst)
        loss = loss_s + self.lambda_u * loss_u

        with torch.no_grad():
            scores = {
                "loss": loss,
                "loss_s": loss_s,
                "loss_u": loss_u,
            }
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            scores_s = self.metric_dict_train_s(self.activation(logits_xs_weak), ys)
            self.log_dict(scores_s, **self.log_params)

            scores_u = self.metric_dict_train_u_pseudo(
                self.activation(logits_xu_weak_lst), yu_lst
            )
            self.log_dict(scores_u, **self.log_params)

        return loss
