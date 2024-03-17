import numpy as np
import cv2


class PaddingTableImage(object):
    """
        pad zeros at the bottom and right of the image
    """

    def __init__(self, size, **kwargs):
        super(PaddingTableImage, self).__init__()
        self.size = size

    def __call__(self, data):
        img = data['image']
        pad_h, pad_w = self.size
        padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        height, width = img.shape[0:2]
        padding_img[0:height, 0:width, :] = img.copy()
        data['image'] = padding_img
        shape = data['shape'].tolist()
        shape.extend([pad_h, pad_w])
        data['shape'] = np.array(shape)
        return data


class ResizeTableImage(object):
    """
        resize that preserve aspect ratio
    """
    def __init__(self, max_len, resize_bboxes=False, infer_mode=False,
                 **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len   # dimension of the longer side after resized
        self.resize_bboxes = resize_bboxes
        self.infer_mode = infer_mode

    def __call__(self, data):
        img = data['image']
        height, width = img.shape[0:2]
        ratio = self.max_len / max(height, width)
        resize_h = int(height * ratio)
        resize_w = int(width * ratio)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        if self.resize_bboxes and not self.infer_mode:
            data['bboxes'] = data['bboxes'] * ratio
        data['image'] = resize_img
        data['src_img'] = img
        data['shape'] = np.array([height, width, ratio, ratio])
        data['max_len'] = self.max_len
        return data



