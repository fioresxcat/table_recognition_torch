import torch
import numpy as np
import pdb
import cv2
from PIL import Image


class DecodeImage:
    """ decode image """

    def __init__(
        self,
        img_mode='RGB',
        channel_first=False,
        ignore_orientation=False,
        **kwargs
    ):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation


    def __call__(self, data):
        img = data['image']
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)

        if img is None:
            return None
        
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data
    


class NormalizeImage(object):
    """ 
        normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        self.scale = np.float32(eval(scale) if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]  # image net mean
        std = std if std is not None else [0.229, 0.224, 0.225]  # image net std

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage: img must be a numpy array"
        data['image'] = (img.astype(np.float32) * self.scale - self.mean) / self.std
        return data
    

class ToCHWImage:
    """ 
        convert hwc image to chw image
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img: np.ndarray = data['image']
        data['image'] = img.transpose((2, 0, 1))
        return data