import numpy as np
import pdb
import os
import random
from torch.utils.data import Dataset, DataLoader
import json
from copy import deepcopy
import logging

from .transforms import DataTranformer


class PubTabDataset(Dataset):
    def __init__(self, config, mode):
        super(PubTabDataset, self).__init__()

        self.global_config = config
        self.config = config.data
        self.mode = mode
        self.anno_path = self.config.dataset[mode].anno_path
        self.data_dir = self.config.dataset[mode].data_dir
        self.data_lines = self.get_image_info_list(self.anno_path)
        self.check(self.global_config.common.max_seq_len)
        self.ops = DataTranformer(self.config.dataset.transforms)


    def get_image_info_list(self, anno_path):
        data_lines = []
        with open(anno_path, "r") as f:
            data_lines = f.readlines()
        data_lines = [line for line in data_lines if not line.startswith("//")]
        return data_lines


    def check(self, max_text_length):
        """
            purpose: remove non-existing image and too long image
        """
        data_lines = []
        for line in self.data_lines:
            data_line = line.strip("\n")
            info = json.loads(data_line)
            file_name = info['filename']
            cells = info['html']['cells'].copy()
            structure = info['html']['structure']['tokens'].copy()

            img_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(img_path):
                self.logger.warning("{} does not exist!".format(img_path))
                continue
            # ignore file with too many tokens
            if len(structure) == 0 or len(structure) > max_text_length:
                continue
            # data = {'img_path': img_path, 'cells': cells, 'structure':structure,'file_name':file_name}
            data_lines.append(line)
        self.data_lines = data_lines


    def __getitem__(self, idx):
        data_line = self.data_lines[idx]
        data_line = data_line.strip("\n")
        info = json.loads(data_line)
        file_name = info['filename']
        cells = info['html']['cells'].copy()
        structure = info['html']['structure']['tokens'].copy()

        img_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(img_path):
            raise Exception("{} does not exist!".format(img_path))
        data = {
            'img_path': img_path,
            'cells': cells,
            'structure': structure,
            'file_name': file_name
        }

        # read image as bytes
        with open(data['img_path'], 'rb') as f:
            img = f.read()
            data['image'] = img

        outs = self.ops(data)

        return outs

    def __len__(self):
        return len(self.data_lines)
