import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from ..backbones import *
from ..necks import *
from ..heads import *


class BaseModel(nn.Module):
    """
        A typical deep learning model with 3 separate components: backbone, neck, head
    """
    def __init__(self, config):
        super().__init__()
        self.global_config = config
        self.config = config.model
        self._build_backbone()
        self._build_neck()
        self._build_head()

    def _build_backbone(self):
        cfg = self.config.backbone
        class_name = cfg.name
        init_args = {k: v for k, v in cfg.items() if k != 'name'}
        self.backbone = eval(class_name)(**init_args)

    def _build_neck(self):
        cfg = self.config.neck
        class_name = cfg.name
        init_args = {k: v for k, v in cfg.items() if k != 'name'}
        self.neck = eval(class_name)(**init_args)
    
    def _build_head(self):
        cfg = self.config.head
        class_name = cfg.name
        # init_args = {k: v for k, v in cfg.items() if k != 'name'}
        self.head = eval(class_name)(self.config)


    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x, labels=labels)
        return x
    
