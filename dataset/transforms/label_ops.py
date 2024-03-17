import numpy as np
import pdb
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        lower=False
    ):

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.pad_str = "pad"
        self.lower = lower

        if character_dict_path is None:
            logging.warning("The character_dict_path is None, model can only recognize number and lower letters")
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            characters = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            characters = list(self.character_str)
        characters = self.add_special_char(characters)
        self.char2idx = {char:i for i, char in enumerate(characters)}
        self.characters = characters


    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_indices = []
        for char in text:
            if char not in self.char2idx:
                continue
            text_indices.append(self.char2idx[char])
        if len(text_indices) == 0:
            return None
        return text_indices



class AttnLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        **kwargs
    ):
        super(AttnLabelEncode, self).__init__(max_text_length, character_dict_path, use_space_char)


    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.pad_str = "pad"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character


    def __call__(self, data):
        text_str = data['label']
        text_indices = self.encode(text_str)
        if text_indices is None:
            return None
        if len(text_indices) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text_indices))
        text_indices = [0] + text_indices + [len(self.characters) - 1] +  \
                       [0] * (self.max_text_len - len(text_indices) - 2)   # add sos and eos, then padding
        data['label'] = np.array(text_indices)
        return data


    def get_ignored_tokens(self):
        beg_idx = np.array(self.char2idx[self.beg_str])
        end_idx = np.array(self.char2idx[self.end_str])
        return [beg_idx, end_idx]


class TableLabelEncode(AttnLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        max_text_length,
        character_dict_path,
        replace_empty_cell_token=False,
        merge_no_span_structure=False,
        learn_empty_box=False,
        loc_reg_num=4,
        **kwargs
    ):
        self.max_text_len = max_text_length
        self.lower = False
        self.learn_empty_box = learn_empty_box
        self.merge_no_span_structure = merge_no_span_structure
        self.replace_empty_cell_token = replace_empty_cell_token

        characters = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                characters.append(line)

        if self.merge_no_span_structure:
            if "<td></td>" not in characters:
                characters.append("<td></td>")
            if "<td>" in characters:
                characters.remove("<td>")

        characters = self.add_special_char(characters)
        self.char2idx = {char:i for i, char in enumerate(characters)}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.characters = characters
        self.loc_reg_num = loc_reg_num
        self.pad_idx = self.char2idx[self.beg_str]   # the ingore index of nn.CrossEntropyLoss
        self.start_idx = self.char2idx[self.beg_str]
        self.end_idx = self.char2idx[self.end_str]

        self.td_token = ['<td>', '<td', '<eb></eb>', '<td></td>']
        self.empty_bbox_token_dict = {
            "[]": '<eb></eb>',   # to make a cell empty, must delete its content in the excel file
            "[' ']": '<eb1></eb1>',
            "['<b>', ' ', '</b>']": '<eb2></eb2>',
            "['\\u2028', '\\u2028']": '<eb3></eb3>',
            "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
            "['<b>', '</b>']": '<eb5></eb5>',
            "['<i>', ' ', '</i>']": '<eb6></eb6>',
            "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
            "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
            "['<i>', '</i>']": '<eb9></eb9>',
            "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']":
            '<eb10></eb10>',
        }

    @property
    def _max_text_len(self):
        return self.max_text_len + 2

    def __call__(self, data):
        cells = data['cells']
        structure = data['structure']
        if self.merge_no_span_structure:
            structure = self._merge_no_span_structure(structure)
        if self.replace_empty_cell_token:
            structure = self._replace_empty_cell_token(structure, cells)
        # remove empty token and add " " to span token
        new_structure = []
        for token in structure:
            if token != '':
                if 'span' in token and token[0] != ' ':
                    token = ' ' + token
                new_structure.append(token)
        # encode structure
        structure = self.encode(new_structure)
        if structure is None:
            return None

        structure = [self.start_idx] + structure + [self.end_idx]  # add sos abd eos
        padding_mask = [1] * len(structure) + [0] * (self._max_text_len - len(structure))  # pad
        structure = structure + [self.pad_idx] * (self._max_text_len - len(structure))  # pad
        structure = np.array(structure)
        padding_mask = np.array(padding_mask)   # mark the padding tokens (not counting the bos and eos token)
        data['structure'] = structure
        data['padding_mask'] = padding_mask

        if len(structure) > self._max_text_len:
            return None

        # encode box
        bboxes = np.zeros((self._max_text_len, self.loc_reg_num), dtype=np.float32)
        bbox_masks = np.zeros((self._max_text_len, 1), dtype=np.float32)
        bbox_idx = 0
        for i, token in enumerate(structure):
            if self.idx2char[token] in self.td_token:  # if this is token for a cell (include empty cell)
                if 'bbox' in cells[bbox_idx] and len(cells[bbox_idx]['tokens']) > 0:  # if this is a non-empty cell
                    bbox = cells[bbox_idx]['bbox'].copy()
                    bbox = np.array(bbox, dtype=np.float32).reshape(-1)
                    bboxes[i] = bbox  # absolute box
                    bbox_masks[i] = 1.0  # 1 means non-empty cell
                if self.learn_empty_box:  # default is false
                    bbox_masks[i] = 1.0
                bbox_idx += 1
                
        data['bboxes'] = bboxes  # shape (max_text_len, 4), with batch it'll have shape (bs, max_text_len, 4)
        data['bbox_masks'] = bbox_masks  # shape (max_text_len, 1), with batch it'll have shape (bs, max_text_len, 1)
        return data


    def _merge_no_span_structure(self, structure):
        """
            This code is refer from:
            https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
            Purpose: replace 2 <td> and </td> tokens with a single <td></td> token
        """
        new_structure = []
        i = 0
        while i < len(structure):
            token = structure[i]
            if token == '<td>':
                token = '<td></td>'
                i += 1  # skip the </td> token
            new_structure.append(token)
            i += 1
        return new_structure


    def _replace_empty_cell_token(self, token_list, cells):
        """
            This fun code is refer from:
            https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
            Purpose: replace label for empty cell from <td> to <eb> (empty bbox)
            -> this is to differentiate non-empty cell from empty cell
            -> this function also shows that the cell order must be inline with the token order
        """

        bbox_idx = 0
        add_empty_bbox_token_list = []
        for token in token_list:
            if token in ['<td></td>', '<td', '<td>']:
                if 'bbox' not in cells[bbox_idx].keys():
                    content = str(cells[bbox_idx]['tokens'])
                    token = self.empty_bbox_token_dict[content]
                add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                add_empty_bbox_token_list.append(token)
        return add_empty_bbox_token_list



class TableBoxEncode(object):
    def __init__(self, in_box_format='xyxy', out_box_format='xyxy', **kwargs):
        assert out_box_format in ['xywh', 'xyxy', 'xyxyxyxy']
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format

    def __call__(self, data):
        img_height, img_width = data['image'].shape[:2]
        bboxes = data['bboxes']
        if self.in_box_format != self.out_box_format:
            if self.out_box_format == 'xywh':
                if self.in_box_format == 'xyxyxyxy':
                    bboxes = self.xyxyxyxy2xywh(bboxes)
                elif self.in_box_format == 'xyxy':
                    bboxes = self.xyxy2xywh(bboxes)

        # normalize
        bboxes[:, 0::2] /= img_width
        bboxes[:, 1::2] /= img_height
        data['bboxes'] = bboxes
        return data

    def xyxyxyxy2xywh(self, bboxes):
        new_bboxes = np.zeros([len(bboxes), 4])
        new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
        new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
        new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
        new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
        return new_bboxes

    def xyxy2xywh(self, bboxes):
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
        new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        return new_bboxes
    