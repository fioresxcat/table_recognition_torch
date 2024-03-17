import pdb
from .generic_ops import KeepKeys
from .img_ops import DecodeImage, NormalizeImage, ToCHWImage
from .label_ops import BaseRecLabelEncode, AttnLabelEncode, TableBoxEncode, TableLabelEncode
from .table_ops import PaddingTableImage, ResizeTableImage
from typing import List, Dict, Any, Union, Optional, Tuple
from easydict import EasyDict


class DataTranformer:
    def __init__(self, transforms: List[Dict]):
        self.transforms = []
        for trans in transforms:
            trans = EasyDict(trans)
            assert len(trans.keys()) == 1
            assert len(trans.values()) == 1
            trans, init_args = list(trans.keys())[0], list(trans.values())[0]
            self.transforms.append(eval(trans)(**init_args) if init_args is not None else eval(trans)())


    def __call__(self, data: dict):
        for trans in self.transforms:
            # pdb.set_trace()
            data = trans(data)
        return data
    

if __name__ == '__main__':
    str = 'DataTranformer'
    obj = eval(str)([1, 2, 3])
    print(obj.transforms)
    pdb.set_trace()