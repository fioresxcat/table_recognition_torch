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
from untitled import *
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
        self.loss = eval(loss_name)(**loss_cfg)


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
        # get training config
        train_cfg = self.config.trainer

        # build model, data, optimizer, and loss
        self.build_model()
        self.build_data()
        self.build_optimizer()
        self.build_loss()

        # prepare things
        best_val_loss = float('inf')
        os.makedirs(train_cfg.ckpt_dir, exist_ok=True)
        self.save_training_info(train_cfg.ckpt_dir)
        self.best_model = None
        
        # train loop
        for ep in range(train_cfg.max_epochs):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                imgs = batch['image']
                structure_logits, loc_preds = self.model(imgs, labels=batch)
                loss = self.loss({'structure_logits': structure_logits, 'loc_preds': loc_preds}, batch)
                total_loss, structure_loss, loc_loss = loss['total_loss'], loss['structure_loss'], loss['loc_loss']
                # gradient descent
                self.optimizer.zero_grad()  # need to zero the gradients because pytorch accumulates them by default
                total_loss.backward()  # compute gradients of the loss w.r.t. the parameters
                self.optimizer.step()  # update the parameters
                self.lr_scheduler.step()  # update the learning rate

                current_lr = self.optimizer.param_groups[0]['lr']
                if i % train_cfg.log_interval == 0:
                    print(f"Epoch {ep}, Batch {i}, LR {current_lr:.4f}, Total Loss {format(total_loss.cpu().item(), '.3f')}, Structure Loss {format(structure_loss.cpu().item(), '.3f')}, Loc Loss {format(loc_loss.cpu().item(), '.3f')}")
            
            # validation loop
            self.model.eval()
            val_loss = 0
            for i, batch in enumerate(self.val_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                imgs = batch['image']
                with torch.no_grad():
                    structure_logits, loc_preds = self.model(imgs)
                loss = self.loss({'structure_logits': structure_logits, 'loc_preds': loc_preds}, batch)
                loss = loss['total_loss']
                val_loss += loss.cpu().item()
            val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {ep}, Val Loss {loss:.3f}")        

            # save checkpoint
            if val_loss < best_val_loss:
                if best_val_loss != float('inf'):
                    os.remove(os.path.join(train_cfg.ckpt_dir, f'best_model-val_loss={best_val_loss:.3f}.pt'))
                best_val_loss = val_loss
                self.best_model = deepcopy(self.model)
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(train_cfg.ckpt_dir, f'best_model-val_loss={val_loss:.3f}.pt')
                )

        # self.infer()


    def infer(self):
        img_fp = 'data/test_imgs/1.png'
        orig_img = cv2.imread(img_fp)
        self.best_model.eval()
        self.build_postprocessor()

        for i, batch in enumerate(self.val_loader):
            imgs = batch['image'].to(self.device)
            with torch.no_grad():
                structure_logits, loc_preds = self.best_model(imgs)
            structure_probs = torch.softmax(structure_logits, dim=2)
            # get loss
            loss = self.loss({'structure_logits': structure_logits, 'loc_preds': loc_preds}, batch)
            total_loss, structure_loss, loc_loss = loss['total_loss'], loss['structure_loss'], loss['loc_loss']
            print(f'total_loss: {total_loss}, structure_loss: {structure_loss}, loc_loss: {loc_loss}')

            result = self.postprocessor({'structure_probs': structure_probs, 'loc_preds': loc_preds}, [orig_img.shape[:2]])
            print(result)

            bb_list = result['bbox_batch_list'][0]
            for bb in bb_list:
                bb = bb.astype(int).tolist()
                cv2.rectangle(orig_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            cv2.imwrite('a.jpg', orig_img)
            print('saved a.jpg')


    def overfit_random_input():
        pass


if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/slanet.yaml")
    args = parser.parse_args()

    # load config
    config = EasyDict(omegaconf.OmegaConf.load(args.config))

    # train
    trainer = BaseTrainer(config)
    trainer.train()