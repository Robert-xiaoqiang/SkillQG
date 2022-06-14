import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from tqdm import tqdm

import re
import os
import json
import time
import abc

from ..helper.TrainHelper import AverageMeter, LoggerPather
from .TrainerModuleCommon import TrainerModuleEvalMixin


class CLSTrainerModule(pl.LightningModule, TrainerModuleEvalMixin):#, metaclass = abc.ABCMeta):
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters(self.config)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        '''
        According to the document of pytorch-lightning, the dataloader will be wrapped with DistributedSampler automatically when using the `DDP/DDP2` acclerator. However, this implementation is also based on the callback mechanism which happens after `trainer.fit(model, datamodule)`. In our construction of the `TrainerModule(pl.LightningModule)`, the dataloader is just passed by the main function and still the vanilla dataloader. We have to divide the len(dataloader) by the #rank_node to obtain the train_batch_per_epoch, which is also known as the number of parallel iterations per epoch under the scenario of multi-processes (GPUs) training.
        By the way, if we do not perform the division, we would obtain the number of sequential iterations per epoch. It is not appliable to the lr_scheduler.
        '''
        self.num_train_batch_per_epoch = len(self.train_dataloader) // len(self.config.DEVICE[5:].split(','))
        self.num_epochs = self.config.TRAIN.NUM_EPOCHS
        self.num_iterations = self.num_epochs * self.num_train_batch_per_epoch

        loggerpather = LoggerPather(self.config)
        # self.logger = loggerpather.get_logger()
        self.log_path = loggerpather.get_log_path()
        self.snapshot_path = loggerpather.get_snapshot_path()
        self.tb_path = loggerpather.get_tb_path()
        self.prediction_path = loggerpather.get_prediction_path()
        # self.writer = SummaryWriter(self.tb_path)

        self.loss_avg_meter = AverageMeter()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.TRAIN.WD,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.config.TRAIN.LR, correct_bias = True)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = self.config.TRAIN.NUM_WARMUP_STEPS, num_training_steps = self.num_iterations
        )

        # Before pl v1.3, lr_scheduler is invoked automatically according to the interval (step or epoch)
        # After pl v1.3, it is a manual operation (manually)
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }

        return [ optimizer ], [ lr_dict ]

    def forward(self, **args):
        return self.model(**args)
    
    def training_step_feeding(self, batch, batch_idx):
        train_inputs, test_inputs, labels_lists = batch
        model_inputs = train_inputs
        outputs = self(**model_inputs)
        loss = outputs['loss']

        return loss

    def training_step_logging(self, batch, batch_idx, loss):
        self.loss_avg_meter.update(loss.item())
        self.log('train/loss_cur', loss.item(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_avg', self.loss_avg_meter.average(), on_step = True, prog_bar = True, logger = False)
        
        iteration = self.current_epoch * self.num_train_batch_per_epoch + batch_idx + 1
        self.logger.experiment.add_scalar('train/loss_cur', loss.item(), iteration)
        self.logger.experiment.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.training_step_feeding(batch, batch_idx)
        bp_loss = self.training_step_logging(batch, batch_idx, loss)
        return bp_loss
  
    def validation_step(self, batch, batch_idx):
        train_inputs, test_inputs, labels_lists, sample_ids = batch
        model_inputs = train_inputs
        with torch.no_grad():
            outputs = self(**model_inputs)
        loss = outputs['loss']
        return loss.item()

    def validation_epoch_end(self, validation_step_outputs):
        loss = sum(validation_step_outputs) / len(validation_step_outputs)

        self.log('val/loss', loss, on_epoch = True, prog_bar = True, logger = False)
        self.logger.experiment.add_scalar('val/loss', loss, self.current_epoch)

    def test_step(self, batch, batch_idx):
        test_inputs, labels_lists, sample_ids = batch
        model_inputs = test_inputs

        with torch.no_grad():
            model_outputs = self(**model_inputs)

        labels_posteriors = F.sigmoid(model_outputs['logits'])
        labels_predictions = (labels_posteriors > 0.5).long().cpu().detach().numpy().tolist()

        labels_posteriors = labels_posteriors.cpu().detach().numpy().tolist()
        labels_lists = labels_lists.cpu().detach().numpy().tolist()

        # B x num_lables, B x num_labels, B x num_labels, B (python list-based)
        return labels_predictions, labels_posteriors, labels_lists, sample_ids

    def test_epoch_end(self, test_step_outputs):
        labels_predictions = [ ]
        labels_posteriors = [ ]
        labels_lists = [ ]
        sample_ids = [ ]

        for batch_labels_predictions, batch_labels_posteriors, batch_labels_lists, batch_sample_ids in test_step_outputs:
            labels_predictions.extend(list(batch_labels_predictions))
            labels_posteriors.extend(list(batch_labels_posteriors))
            labels_lists.extend(list(batch_labels_lists))
            sample_ids.extend(list(batch_sample_ids))

        subdir = 'best'
        average_dict = self.write_rank_prediction(labels_predictions, labels_lists, labels_posteriors, sample_ids, subdir)

        self.save_huggingface_model('huggingface_best')

        # wait for other processes and gather the distributed results using rank 0 node / process / GPU
        if not torch.distributed.get_rank():
            time.sleep(1)
            self.gather_rank_prediction(subdir)

        return average_dict
