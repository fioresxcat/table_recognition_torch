from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import pdb


# import paddle
# from paddle import nn
# from paddle.nn import functional as F


class SLALoss(nn.Module):
    def __init__(self, structure_weight, loc_weight, loc_loss, **kwargs):
        super(SLALoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='mean')
        # self.structure_weight = structure_weight / (structure_weight + loc_weight)
        # self.loc_weight = loc_weight / (structure_weight + loc_weight)
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.loc_loss = loc_loss
        self.eps = 1e-12


    def forward(self, predicts, batch):
        """
            predicts: dict with 2 keys
                structure_probs: (bs, max_text_length, num_rec_classes)
                loc_preds: (bs, max_text_length, 4)
            batch: dict with 5 keys
                image: (bs, 3, h, w)
                structure: (bs, max_text_length+1)
                bboxes: (bs, max_text_length+1, 4)
                bbox_masks : (bs, max_text_length+1, 1)  (maybe for cell without text, only calculate loc loss for cell with text inside)
                ??? : (bs, 6)
        """
        structure_logits = predicts['structure_logits']
        structure_targets = batch['structure'].to(torch.int64)
        padding_masks = batch['padding_mask'].to(torch.int64)
        padding_masks = padding_masks[:, 1:]
        structure_targets = structure_targets[:, 1:]  # (bs, max_text_length)
        structure_targets = torch.where(padding_masks == 1, structure_targets, -100)        
        structure_loss = self.loss_func(
            rearrange(structure_logits, 'b l d -> (b l) d'),   # batch, length, dim
            rearrange(structure_targets, 'b l -> (b l)')
        )

        loc_preds = predicts['loc_preds']
        loc_targets = batch['bboxes'].to(torch.float32)
        loc_targets_mask = batch['bbox_masks'].to(torch.float32)
        loc_targets = loc_targets[:, 1:, :]  # (bs, max_text_length, 4)
        loc_targets_mask = loc_targets_mask[:, 1:, :]  # (bs, max_text_length, 1)

        # only calculate loc loss for cell with text inside
        loc_loss = F.l1_loss(
            rearrange(loc_preds, 'b l d -> (b l) d') * rearrange(loc_targets_mask, 'b l d -> (b l) d'),
            rearrange(loc_targets, 'b l d -> (b l) d') * rearrange(loc_targets_mask, 'b l d -> (b l) d'),
            reduction='sum'
        )
        # calculate mean loss fox each token box, only count cells with text inside
        loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)

        total_loss = self.structure_weight * structure_loss + self.loc_weight * loc_loss
        return {
            'total_loss': total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss
        }
