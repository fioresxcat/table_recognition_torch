import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from einops import rearrange, repeat, reduce


class InfoNCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config.temperature


    def forward(self, fea, hidden_html):
        """
            fea: the roi-pooled vectors extract from output of backbone, shape, (num_nc, d_model)
            hidden_html: hidden html output from model, shape (num_nc, d_model)
        """
        # compute the similarity matrix
        num_rois, d_model = fea.shape
        fea = F.normalize(fea, dim=-1)
        hidden_html = F.normalize(hidden_html, dim=-1)
        sim_matrix = torch.matmul(fea, hidden_html.transpose(0, 1))  # shape (num_rois, num_rois)
        sim_matrix = sim_matrix / self.temperature # temperature
        labels = torch.arange(num_rois).to(sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss



class VastLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.html_loss = nn.CrossEntropyLoss()
        self.coord_loss = nn.CrossEntropyLoss()
        self.html_weight = config.html_weight
        self.coord_weight = config.coord_weight
        self.visual_align_weight = config.visual_align_weight
        self.infoNCE_loss = InfoNCELoss(config.infoNCE_loss)


    def forward(self, pred, batch):
        """
            fea: feature map output from backbone
            pred: output from model
            batch: returned from dataloader
        """
        html_hidden = pred['html_hidden']  # shape (bs, max_seq_len, d_model)
        pred_htmls = pred['html_out']  # logits output, shape (bs, max_seq_len, vocab_size)
        pred_coords = pred['coord_out']  # logits output, shape (bs, max_seq_len, 4, n_bins)
        roi_features = pred['roi_features']  # shape (num_nc_in_the_batch, d_model)
        padding_masks = batch['padding_mask']  # shape (bs, max_seq_len)
        bbox_masks = batch['bbox_masks']  # shape (bs, max_seq_len, 1)

        # --------------------------- html loss ---------------------------
        # batch['html_label'] shape: (bs, max_seq_len)
        labels = batch['structure']
        padding_masks = padding_masks[:, 1:]
        labels = labels[:, 1:]
        labels = torch.where(padding_masks == 1, labels, -100)  # replace the padding with -100 for the cross entropy loss to ignore the padding
        html_loss = self.html_loss(
            rearrange(pred_htmls, 'b l d -> (b l) d'), 
            rearrange(labels, 'b l -> (b l)')
        )

        # ---------------------------- coord loss --------------------------------
        # batch['coord_label'] shape: (bs, max_seq_len, num_coords)
        coords = batch['abs_bboxes'][:, 1:, :].long()  # shape (bs, max_seq_len+1, 4)
        coord_masks = batch['bbox_masks'][:, 1:, :] # shape (bs, max_seq_len+1, 1)
        coord_masks = repeat(coord_masks, 'b l c -> b l (c d)', d=4) # shape (bs, max_seq_len+1, 4)
        coords_flatten = coords.reshape(-1)
        coords_flatten = torch.where(coord_masks.reshape(-1) == 1, coords_flatten, -100)
        coord_loss = self.coord_loss(
            rearrange(pred_coords, 'b l c d -> (b l c) d'),   # shape (bs*max_seq_len*num_coords, n_bins)
            coords_flatten # shape (bs*max_seq_len*num_coords)
        )

        # ----------------------------- visual-alignment loss --------------------------
        if roi_features is not None:
            nc_indices = bbox_masks.view(-1).nonzero().squeeze()  # nc means non-empty cells
            nc_html_hidden = rearrange(html_hidden, 'b l d -> (b l) d')[nc_indices]  # shape (num_nc_in_the_batch, d_model)
            visual_align_loss = self.infoNCE_loss(roi_features, nc_html_hidden)
        else:
            visual_align_loss = 0

        # total loss
        loss = self.html_weight * html_loss + self.coord_weight * coord_loss + self.visual_align_weight * visual_align_loss
        return {
            'html_loss': html_loss,
            'coord_loss': coord_loss,
            'visual_align_loss': visual_align_loss,
            'total_loss': loss
        }
        