import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from einops import rearrange, repeat, reduce


class InfoNCELoss(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.temperature = config.temperature
        pass

    def forward(self, fea, hidden_html):
        """
            fea: the roi-pooled vectors extract from output of backbone, shape, (bs, num_rois, d_model)
            hidden_html: hidden html output from model, shape (bs, num_rois, d_model)
        """
        # compute the similarity matrix
        bs, num_rois, d_model = fea.shape
        # fea = F.normalize(fea, dim=-1)
        # hidden_html = F.normalize(hidden_html, dim=-1)
        sim_matrix = torch.matmul(fea, hidden_html.transpose(1, 2))  # shape (bs, num_rois, num_rois)

        # compute the loss
        sim_matrix = sim_matrix / self.temperature # temperature
        pos = torch.diag(sim_matrix)  # shape (bs, num_rois)
        neg = torch.exp(sim_matrix - pos.unsqueeze(1))  # shape (bs, num_rois, num_rois)
        neg = neg.sum(dim=-1)  # shape (bs, num_rois)
        loss = -torch.log(pos / neg)
        loss = loss.mean()
        return loss


class VastLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.html_loss = nn.CrossEntropyLoss()
        self.coord_loss = nn.CrossEntropyLoss()
        self.html_weight = config.html_weight
        self.coord_weight = config.coord_weight
        self.visual_align_weight = config.visual_align_weight
        self.infoNCE_loss = InfoNCELoss()

    def forward(self, fea, pred, batch):
        """
            fea: feature map output from backbone
            pred: output from model
            batch: returned from dataloader
        """
        html_hidden = pred['html_hidden']  # shape (bs, max_seq_len, d_model)
        pred_htmls = pred['html_out']  # logits output, shape (bs, max_seq_len, vocab_size)
        pred_coords = pred['coord_out']  # logits output, shape (bs, max_seq_len, num_coords, n_bins)
        # --------------------------- html loss ---------------------------
        # batch['html_label'] shape: (bs, max_seq_len)
        html_loss = self.html_loss(
            rearrange(pred_htmls, 'b l d -> (b l) d'), 
            rearrange(batch['html_label'], 'b l -> (b l)')
        )

        # ---------------------------- coord loss --------------------------------
        # batch['coord_label'] shape: (bs, max_seq_len, num_coords)
        coord_loss = self.coord_loss(
            rearrange(pred_coords, 'b l c d -> (b l c) d'), 
            rearrange(batch['coord_label'], 'b l c -> (b l c)')
        )

        # ----------------------------- visual-alignment loss --------------------------
        visual_align_loss = self.infoNCE_loss(fea, html_hidden)

        # total loss
        loss = self.html_weight * html_loss + self.coord_weight * coord_loss + self.visual_align_weight * visual_align_loss
        return loss
        