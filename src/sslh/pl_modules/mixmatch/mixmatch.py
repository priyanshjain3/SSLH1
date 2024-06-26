#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import torch

from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from sslh.nn.loss import CrossEntropyLossVecTargets
from sslh.nn.utils import ForwardDictAffix
from sslh.transforms.other.mixup import MixupModule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
class MixMatch(LightningModule):
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
        alpha: float = 0.25,
        train_metrics: Optional[Dict[str, nn.Module]] = None,
        val_metrics: Optional[Dict[str, nn.Module]] = None,
        log_on_epoch: bool = True,
    ) -> None:
        """
        MixMatch (MM) LightningModule.

        :param model: The PyTorch nn.Module to train.
                The forward() must return logits to classify the data.
        :param optimizer: The PyTorch optimizer to use.
        :param activation: The activation function of the model.
                (default: nn.Softmax(dim=-1))
        :param criterion_s: The loss component 'L_s' of MM.
                (default: CrossEntropyWithVectors())
        :param criterion_u: The loss component 'L_u' of MM.
                (default: CrossEntropyWithVectors())
        :param lambda_u: The coefficient of the 'L_u' component. (default: 1.0)
        :param n_augms: The number of strong augmentations applied. (default: 2)
        :param temperature: The temperature applied by the sharpen function.
                A lower temperature make the pseudo-label produced more 'one-hot'.
                (default: 0.5)
        :param alpha: The mixup alpha parameter. A higher value means a stronger mix between labeled and unlabeled data.
                (default: 0.25)
        :param train_metrics: An optional dictionary of metrics modules for training.
                (default: None)
        :param val_metrics: An optional dictionary of metrics modules for validation.
                (default: None)
        :param log_on_epoch: If True, log only the epoch means of each train metric score.
                (default: True)
        """
        super().__init__()
        self.model = model
        self.activation = activation
        self.optimizer = optimizer
        self.criterion_s = criterion_s
        self.criterion_u = criterion_u
        self.lambda_u = lambda_u
        self.n_augms = n_augms
        self.temperature = temperature
        self.alpha = alpha
        self.num_classes=10
        self.metric_dict_train_s = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_s"
        )
        self.metric_dict_train_u_pseudo = ForwardDictAffix(
            train_metrics, prefix="train/", suffix="_u"
        )
        self.metric_dict_val = ForwardDictAffix(val_metrics, prefix="val/")
        self.metric_dict_test = ForwardDictAffix(val_metrics, prefix="test/")

        self.log_params = dict(on_epoch=log_on_epoch, on_step=not log_on_epoch)
        self.mixup = MixupModule(alpha=alpha, apply_max=True)

        self.save_hyperparameters(
            {
                "experiment": self.__class__.__name__,
                "model": model.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "activation": activation.__class__.__name__,
                "criterion_s": criterion_s.__class__.__name__,
                "criterion_u": criterion_u.__class__.__name__,
                "lambda_u": lambda_u,
                "n_augms": n_augms,
                "temperature": temperature,
                "alpha": alpha,
            }
        )

    def training_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], List[Tensor]],
        batch_idx: int,
    ) -> Tensor:
        (xs_weak, ys), xu_weak_lst = batch
        probs_xs = self(xs_weak)
        with torch.no_grad():
            # Guess pseudo-label 'yu' and repeat
            yu = self.guess_label(xu_weak_lst)
            yu_lst = yu.repeat([self.n_augms] + [1] * (len(yu.shape) - 1))

            # Stack augmented 'xu' variants to a single batch
            xu_weak_lst = torch.vstack(xu_weak_lst)

            xs_weak_mix, xu_weak_mix, ys_mix, yu_mix = self.mixmatch(
                xs_weak, xu_weak_lst, ys, yu_lst
            )

        logits_xs_mix = self.model(xs_weak_mix)
        logits_xu_mix = self.model(xu_weak_mix)

        loss_s = self.criterion_s(logits_xs_mix, ys_mix)
        loss_u = self.criterion_u(logits_xu_mix, yu_mix)
        loss = loss_s + self.lambda_u * loss_u

        with torch.no_grad():
            scores = {
                "loss": loss,
                "loss_s": loss_s,
                "loss_u": loss_u,
            }
            scores = {f"train/{k}": v.cpu() for k, v in scores.items()}
            self.log_dict(scores, **self.log_params)

            probs_xs_weak = self.activation(self.model(xs_weak))
            scores_s = self.metric_dict_train_s(probs_xs_weak, ys)
            self.log_dict(scores_s, **self.log_params)

            probs_xu_weak_lst = self.activation(self.model(xu_weak_lst))
            scores_u = self.metric_dict_train_u_pseudo(probs_xu_weak_lst, yu_lst)
            self.log_dict(scores_u, **self.log_params)
        #preds_one_hot = torch.zeros_like(probs_xs)
        #preds_one_hot[torch.arange(probs_xs.size(0)), torch.argmax(probs_xs, dim=1)] = 1

        #print(preds_xs.shape)
        # Update confusion matrix (assuming it's initialized as self.confusion_matrix_val)
        #for t, p in zip(ys.cpu().numpy(), preds_one_hot.cpu().numpy()):
            #self.confusion_matrix_val += confusion_matrix(t, p, labels=list(range(self.num_classes)))
        #self.visualize_confusion_matrix(self.confusion_matrix_val, title='Validation Confusion Matrix',filename='train_confusion_matrix.png')
        return loss

    def mixmatch(
        self,
        xs_weak: Tensor,
        xu_weak_lst: Tensor,
        ys: Tensor,
        yu_lst: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Apply Mixup between labeled and unlabeled data.
        Note: xs_weak and xu_weak_lst must have the same number of dimension but they can have a different bsize.

        :param xs_weak: (bsize_s, *features...)
        :param xu_weak_lst: (bsize_u*n_augms, *features...)
        :param ys: (bsize_s, n_classes)
        :param yu_lst: (bsize_u*n_augms, n_classes)
        :return: The tuple of labeled and unlabeled data mixed : (xs_mixed, xu_mixed, ys_mixed, yu_mixed).
        """
        # Prepare W
        xw = torch.cat((xs_weak, xu_weak_lst))
        yw = torch.cat((ys, yu_lst))

        # Shuffle W
        indices = torch.randperm(len(xw))
        xw, yw = xw[indices], yw[indices]
        alpha=0.5
        
        self.log_params = dict(on_epoch=True, on_step=not True)
        self.mixup = MixupModule(alpha=alpha, apply_max=True)
        bsize_s = len(xs_weak)
        xs_mix, ys_mix = self.mixup(xs_weak, xw[:bsize_s], ys, yw[:bsize_s])
        xu_mix, yu_mix = self.mixup(xu_weak_lst, xw[bsize_s:], yu_lst, yw[bsize_s:])
        #print("Mixup applied for alpha = ",alpha)
        return xs_mix, xu_mix, ys_mix, yu_mix

    @torch.no_grad()
    def guess_label(self, xu_weak_lst: List[Tensor]) -> Tensor:
        assert len(xu_weak_lst) > 0
        probs_xu_weak_lst = self.activation(self.model(xu_weak_lst[0]))
        for xu_weak in xu_weak_lst[1:]:
            probs_xu_weak_lst += self.activation(self.model(xu_weak))
        probs_xu_weak_lst /= self.n_augms
        yu = self.sharpen(probs_xu_weak_lst)
        return yu

    def sharpen(self, pred: Tensor) -> Tensor:
        pred = pred ** (1.0 / self.temperature)
        pred = pred / pred.norm(p=1, dim=-1, keepdim=True)  # type: ignore
        return pred
    def on_validation_epoch_start(self) -> None:
        self.confusion_matrix_val = np.zeros((self.num_classes, self.num_classes), dtype=np.int)

    def on_test_epoch_start(self) -> None:
        self.confusion_matrix_test = np.zeros((self.num_classes, self.num_classes), dtype=np.int)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        xs, ys = batch
        #print(ys.shape)
        probs_xs = self(xs)
        preds_xs = torch.argmax(probs_xs, dim=1)
        if len(ys.shape) > 1 and ys.shape[1] > 1:
            ys = torch.argmax(ys, dim=1)
        # preds_one_hot = torch.zeros_like(probs_xs)
        # preds_one_hot[torch.arange(probs_xs.size(0)), torch.argmax(probs_xs, dim=1)] = 1

        #print(preds_xs.shape)
        # Update confusion matrix (assuming it's initialized as self.confusion_matrix_val)
        self.confusion_matrix_val += confusion_matrix(ys.cpu().numpy(), preds_xs.cpu().numpy(), labels=list(range(self.num_classes)))
        #self.confusion_matrix_val += confusion_matrix(ys.cpu().numpy(), preds_xs.cpu().numpy(), labels=list(range(self.num_classes)))
        #self.visualize_confusion_matrix(self.confusion_matrix_val, title='Validation Confusion Matrix',filename='val_confusion_matrix.png')
    
        self.log_dict(
            self.metric_dict_val(probs_xs, ys),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        xs, ys = batch
        probs_xs = self(xs)
        preds_xs = torch.argmax(probs_xs, dim=1)
        if len(ys.shape) > 1 and ys.shape[1] > 1:
            ys = torch.argmax(ys, dim=1)
        # preds_one_hot = torch.zeros_like(probs_xs)
        # preds_one_hot[torch.arange(probs_xs.size(0)), torch.argmax(probs_xs, dim=1)] = 1

        #print(preds_xs.shape)
        # Update confusion matrix (assuming it's initialized as self.confusion_matrix_val)
        self.confusion_matrix_val += confusion_matrix(ys.cpu().numpy(), preds_xs.cpu().numpy(), labels=list(range(self.num_classes)))
        #self.confusion_matrix_test += confusion_matrix(ys.cpu().numpy(), preds_xs.cpu().numpy(), labels=list(range(self.num_classes)))
        #self.visualize_confusion_matrix(self.confusion_matrix_test, title='Test Confusion Matrix',filename='test_confusion_matrix.png') 
        self.log_dict(
            self.metric_dict_test(probs_xs, ys),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
    def visualize_confusion_matrix(self, cm, title='Confusion Matrix', filename='confusion_matrix.png'):
        matplotlib.use('Agg')
        plt.ioff()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(filename)
        print(f"Confusion matrix saved to {filename}")
        plt.close()
    def on_validation_epoch_end(self) -> None:
        self.visualize_confusion_matrix(self.confusion_matrix_val, title='Validation Confusion Matrix',filename='val_confusion_matrix.png')
    def on_test_epoch_end(self) -> None:
        self.visualize_confusion_matrix(self.confusion_matrix_test, title='Test Confusion Matrix',filename='test_confusion_matrix.png')
    
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.model(x))

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer
