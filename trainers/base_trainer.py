import torch
import cv2
import os
import numpy as np
import pdb
import logging
import argparse
from typing import List, Tuple, Dict, Any, Union, Optional
from easydict import EasyDict
import omegaconf
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.architectures.base_model import BaseModel
from dataset.pubtab_dataset import PubTabDataset
from losses import *
from torch.optim import AdamW
from postprocess.table_postprocess import TableLabelDecode
from utils import *
from copy import deepcopy
import matplotlib.pyplot as plt


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.trainer.device


    def build_model(self):
        self.model = BaseModel(self.config)
        if self.config.model.pretrained is not None:
            state_dict = torch.load(self.config.model.pretrained, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print(f'loaded pretrained weight successfully')
        
        self.model.to(self.device)


    def build_data(self):
        self.train_ds = PubTabDataset(self.config, mode='train')
        self.val_ds = PubTabDataset(self.config, mode='val')
        loader_cfg = self.config.data.loader
        self.train_loader = DataLoader(self.train_ds, **loader_cfg)
        self.val_loader = DataLoader(self.val_ds, **loader_cfg)

        # item = self.train_ds[0]
        

    def build_optimizer(self):
        opt_name = self.config.optimizer.name
        opt_cfg = {k:v for k, v in self.config.optimizer.items() if k != 'name'}
        self.optimizer = torch.optim.__dict__[opt_name](
            self.model.parameters(), **opt_cfg
        )

        scheduler_name = self.config.scheduler.name
        scheduler_cfg = {k:v for k, v in self.config.scheduler.items() if k != 'name'}
        self.lr_scheduler = torch.optim.lr_scheduler.__dict__[scheduler_name](
            self.optimizer, **scheduler_cfg
        )

    def build_loss(self):
        loss_name = self.config.loss.name
        loss_cfg = {k:v for k, v in self.config.loss.items() if k != 'name'}
        self.loss = eval(loss_name)(self.config.loss)


    def build_postprocessor(self):
        self.postprocessor = TableLabelDecode(
            character_dict_path='data/table_structure_dict.txt',
            merge_no_span_structure=True
        )


    def save_training_info(self, save_dir):
        shutil.copy(self.config.data.dataset.train.anno_path, save_dir)
        shutil.copy(self.config.data.dataset.val.anno_path, save_dir)
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            omegaconf.OmegaConf.save(dict(self.config), f)

    
    def train(self):
        raise NotImplementedError('train method must be implemented')
    

    def infer(self):
        raise NotImplementedError('infer method must be implemented')

        # self.infer()


    def overfit_random_input():
        pass