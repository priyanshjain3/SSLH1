#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import confusion_matrix
import numpy as np
class FlushLoggerCallback(Callback):
    def _flush_pl(self, pl_module: LightningModule) -> None:
        if not isinstance(pl_module.logger, TensorBoardLogger):
            return None

        experiment = pl_module.logger.experiment
        if not isinstance(experiment, SummaryWriter):
            raise TypeError(
                f"Unknown experiment type {type(experiment)=} for FlushLoggerCallback."
            )

        experiment.flush()

