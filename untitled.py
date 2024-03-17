import os
import cv2
import numpy as np
import pdb
import shutil
from pathlib import Path
from PIL import Image
import json


def to_abs_coord(coords, img_shape):
    if isinstance(coords, list):
        coords = np.array(coords)
    img_h, img_w = img_shape
    coords = coords.copy()
    coords[::2] = coords[::2] * img_w
    coords[1::2] = coords[1::2] * img_h
    coords = coords.astype(int).tolist()
    return coords


def parse_jsonl(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    data = [json.loads(l) for l in lines]
    return data

def nothing():
    img = cv2.imread('data/test_imgs/1.png')
    fp = 'data/test_imgs/anno_train.jsonl'
    data = parse_jsonl(fp)
    data = data[0]
    bboxes = [cell['bbox'] for cell in data['html']['cells']]
    for bb in bboxes:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    cv2.imwrite('a.jpg', img)


if __name__ == '__main__':
    nothing()
