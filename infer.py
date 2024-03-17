import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import omegaconf
from easydict import EasyDict
from models.architectures.base_model import BaseModel
from dataset.transforms import DataTranformer, DecodeImage, NormalizeImage, ResizeTableImage, PaddingTableImage, ToCHWImage
from postprocess.table_postprocess import TableLabelDecode


def main(args):
    config = EasyDict(omegaconf.OmegaConf.load(args.config))
    model = BaseModel(config)
    model.load_state_dict(torch.load(args.ckpt))
    print(f'loaded model from {args.ckpt} successfully')
    model.eval().to(args.device)

    # prepare data transformation
    data_transformer = DataTranformer(config.data.dataset.transforms)
    new_trans = []
    for trans in data_transformer.transforms:
        if 'image' not in type(trans).__name__.lower():
            continue
        new_trans.append(trans)
    data_transformer.transforms = new_trans

    # prepare post processor
    postprocessor = TableLabelDecode(
        character_dict_path='data/table_structure_dict.txt',
        merge_no_span_structure=True
    )

    # infer
    if os.path.isfile(args.img_path):
        img_paths = [args.img_path]
    elif os.path.isdir(args.img_path):
        img_paths = sorted(list(Path(args.img_path).rglob('*.jpg')))
    for fp in img_paths:
        data = {}
        with open(fp, 'rb') as f:
            img = f.read()
            data['image'] = img
        orig_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        data = data_transformer(data)
        img = data['image']
        imgs = torch.from_numpy(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            structure_logits, loc_preds = model(imgs)
        structure_logits = structure_logits.cpu()
        loc_preds = loc_preds.cpu()
        print(structure_logits.shape, loc_preds.shape)  # b, l, d
        structure_probs = torch.softmax(structure_logits, dim=2)
        # decode the structure_probs and loc_preds
        result = postprocessor(
            {
                'structure_probs': structure_probs,
                'loc_preds': loc_preds
            },
            [orig_img.shape[:2]]
        )
        print(result)
        bb_list = result['bbox_batch_list'][0]
        for bb in bb_list:
            bb = bb.astype(int).tolist()
            cv2.rectangle(orig_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.imwrite('a.jpg', orig_img)
        print('saved a.jpg')



if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/slanet.yaml")
    parser.add_argument("--ckpt", type=str, default="path/to/ckpt")
    parser.add_argument("--img_path", type=str, default="path/to/img_dir")
    parser.add_argument("--device", type=str, default="cuda:7")
    args = parser.parse_args()

    main(args)