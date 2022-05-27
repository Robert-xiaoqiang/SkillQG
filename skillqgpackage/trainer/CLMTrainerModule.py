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

from .BaseTrainerModule import BaseTrainerModule
from .TrainerModuleCommon import TrainerModuleCLMMixin

class CLMTrainerModule(BaseTrainerModule, TrainerModuleCLMMixin):
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, config):
        super().__init__(model, tokenizer, train_dataloader, val_dataloader, config)
