import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from copy import deepcopy
from einops import rearrange, repeat, reduce
from easydict import EasyDict
import math
from copy import deepcopy


class ResBlock(nn.Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, orig_inp, layer_inp):
        layer_out = self.layer(layer_inp)
        out = self.layer_norm(layer_out + orig_inp)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, inp_dim=128, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)   # hidden_dim often bigger than input dim (expansion)
        self.fc2 = nn.Linear(hidden_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            a feed forward network that is applied to each token seperately and identically
            contain 2 linear layers with a ReLU activation in between
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=384, n_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=True,
            batch_first=True
        )
        
    def forward(self, inp):
        q, k, v, key_padding_mask, attn_mask = inp
        # repeat attn mask for each head
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.mha.num_heads, 1, 1)
        attn_output, attn_output_weights = self.mha(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return attn_output
    


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: shape (batch_size, seq_len, d_model)
        """
        try:
            x = x + self.pe[:x.size(1), :]
        except Exception as e:
            raise e
            print(e)
            pdb.set_trace()
        return self.dropout(x)
    



class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        mha = MultiHeadAttention(d_model=config.d_model, n_heads=config.n_heads, dropout=config.dropout)
        ffn = FeedForwardNetwork(inp_dim=config.d_model, hidden_dim=config.ffn_hidden_dim, dropout=config.dropout)
        self.resblock1 = ResBlock(mha, config.d_model)   # masked self mha
        self.resblock2 = ResBlock(deepcopy(mha), config.d_model)   # cross mha
        self.resblock3 = ResBlock(ffn, config.d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.resblock1(orig_inp=x, layer_inp=(x, x, x, None, tgt_mask))
        x = self.resblock2(orig_inp=x, layer_inp=(x, enc_out, enc_out, src_mask, None))
        x = self.resblock3(orig_inp=x, layer_inp=x)
        return x
    


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
            x: shape (batch_size, seq_len): token indices in the vocabulary
        """
        return self.embedding(x) * math.sqrt(self.d_model)



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(d_model=config.d_model, max_seq_len=config.max_seq_len+1)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
    
    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
            query: x, shape (batch_size, max_seq_len+1, d_model)
            key, value: enc_out
            src_mask: mask for key, value
            tgt_mask: causal mask for query
        """
        x = self.pe(x)  # add positional encoding to the input
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        hidden = x
        out = self.classifier(x)
        return hidden, out
    
    

class VastHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.global_config = config
        self.config = config.model.head
        self.fea_pe = SinusoidalPositionalEncoding(d_model=self.config.d_model, max_seq_len=38*38)
        self.html_emb = TokenEmbedding(vocab_size=self.config.vocab_size, d_model=self.config.d_model)
        self.html_decoder = Decoder(config=self.config.html_decoder)
        self.coord_emb = TokenEmbedding(vocab_size=self.config.n_bins, d_model=self.config.d_model)
        self.coord_decoder = Decoder(config=self.config.coord_decoder)



    def forward(self, inputs, labels=None):
        """
            fea: feature map output of the neck module
            img is resized to 608 x 608, and the feature map size is 1/16 of the input image
            -> feature map size = 38 x 38
        """
        # flatten the feature map and and position encoding
        html_out, coord_out, html_hidden = None, None, None

        fea = deepcopy(inputs[0])
        bs = fea.size(0)  # get the batch size
        fea = rearrange(fea, 'b c h w -> b (h w) c')
        fea = self.fea_pe(fea)

        if self.train and labels is not None:
            # compute html output
            html_indices = labels['html_target'][:, :-1]  # shape (batch_size, max_seq_len+1)
            html_emb = self.html_emb(html_indices)  # shape (batch_size, max_seq_len+1, d_model)
            causal_mask = torch.triu(torch.ones((html_indices.size(0), html_indices.size(1), html_indices.size(1))), diagonal=1).bool().to(fea.device)
            html_hidden, html_out = self.html_decoder(x=html_emb, enc_out=fea, src_mask=None, tgt_mask=causal_mask)  # shape (batch_size, max_seq_len+1, d_model)
            # compute box output
            coord_indices = labels['coord_target'][:, :, :-1]  # shape (batch_size, max_seq_len + 1, 3)
            coord_indices = rearrange(coord_indices, 'b l d -> (b l) d')  # shape (batch_size * (max_seq_len+1), 3)
            coord_emb = self.coord_emb(coord_indices)  # shape (batch_size * (max_seq_len+1), 3, d_model)
            coord_emb = torch.concat([rearrange(html_hidden, 'b l d -> (b l) 1 d'), coord_emb], dim=1)  # shape (bs*(max_seq_len+1), 4, d_model)
            fea_repeat = repeat(fea, 'b l d -> (b n) l d', n=coord_emb.size(0) // fea.size(0))   # n = max_seq_len, shape (batch_size * max_seq_len, h*w, d_model)
            causal_mask = torch.triu(torch.ones((coord_emb.size(0), coord_emb.size(1), coord_emb.size(1))), diagonal=1).bool().to(fea.device)  # shape (bs*max_seq_len, 4, 4)
            # query: coord_emb, shape (batch_size * max_seq_len, 4, d_model)
            # key: fea_repeat, shape (batch_size * max_seq_len, h*w, d_model)
            # value: fea_repeat, shape (batch_size * max_seq_len, h*w, d_model)
            # src_mask: none, because we want to attend to all the tokens in the feature map
            # tgt_mask: causal mask, because we want to hide the future tokens in coord_emb
            coord_hidden, coord_out = self.coord_decoder(coord_emb, enc_out=fea_repeat, src_mask=None, tgt_mask=None)  # shape (bs * max_seq_len, 4, d_model), (bs * max_seq_len, 4, n_bins)
            coord_out = rearrange(coord_out, '(bs maxlen) num_coord nbins -> bs maxlen num_coord nbins', bs=bs)  # shape (bs, max_seq_len, 4, n_bins)
            # when compute loss will reshape coord_out to (batch_size * max_seq_len * 4, n_bins)
            # and the label return from data loader will have shape (batch_size * max_seq_len * 4)
            # then feed the two in the cross entropy loss function

        else:  # auto-regressive decoding
            # decode html sequence
            pre_chars = torch.zeros((bs, 1), dtype=torch.long).to(fea.device)
            for i in range(self.config.max_seq_len+1):
                emb = self.html_emb(pre_chars)  # shape (bs, 1, d_model)
                html_hidden, html_out = self.html_decoder(x=emb, enc_out=fea, src_mask=None, tgt_mask=None)  # shape (bs, i+1, d_model)
                cur_chars = torch.argmax(html_out, dim=-1)[:, -1:]  # shape (batch_size, 1)
                pre_chars = torch.concat([pre_chars, cur_chars], dim=1)  # shape (batch_size, i+2)
            # decode coord sequence
            init_coord_emb = rearrange(html_hidden, 'b l d -> (b l) 1 d')  # shape (bs*max_seq_len, 1, d_model)
            pre_coords = torch.empty(bs*html_out.shape[1], 0, dtype=torch.long)
            fea_repeat = repeat(fea, 'b l d -> (b n) l d', n=init_coord_emb.size(0) // fea.size(0))   # n = max_seq_len, shape (batch_size * max_seq_len, h*w, d_model)
            for i in range(4):
                if pre_coords.shape[1] == 0:
                    pre_coords_emb = init_coord_emb
                else:
                    pre_coords_emb = self.coord_emb(pre_coords)  # shape (bs*max_seq_len, i, d_model)
                    pre_coords_emb = torch.concat([init_coord_emb, pre_coords_emb], dim=1)  # shape (bs*max_seq_len, i+1, d_model)
                coord_hidden, coord_out = self.coord_decoder(x=pre_coords_emb, enc_out=fea_repeat, src_mask=None, tgt_mask=None)   # shape (bs*max_seq_len, 1, d_model)
                coord = coord_out.argmax(dim=-1)[:, -1:]  # shape (bs*max_seq_len, 1)
                pre_coords = torch.concat([pre_coords, coord], dim=1)  # shape (bs*max_seq_len, i+2)
            coord_out = rearrange(coord_out, '(bs maxlen) num_coord nbins -> bs maxlen num_coord nbins', bs=bs)
            
        return {
            'html_out': html_out,
            'coord_out': coord_out,
            'html_hidden': html_hidden,
            'fea': inputs[0],
        }
    


if __name__ == '__main__':
    config = {
        'model': {
            'head': {
                'd_model': 512,
                'max_seq_len': 8,
                'vocab_size': 30,
                'n_bins': 608,
                'html_decoder': {
                    'd_model': 512,
                    'max_seq_len': 8,
                    'n_layers': 3,
                    'n_heads': 8,
                    'ffn_hidden_dim': 512,
                    'dropout': 0.1,
                    'num_classes': 30,

                },
                'coord_decoder': {
                    'd_model': 512,
                    'max_seq_len': 8,
                    'n_layers': 3,
                    'n_heads': 8,
                    'ffn_hidden_dim': 512,
                    'dropout': 0.1,
                    'num_classes': 608,

                }
            }
        }
    }
    config = EasyDict(config)
    head = VastHead(config)

    # prepare input
    bs = 2
    fea = torch.rand(bs, 512, 38, 38)
    html_target = torch.concat([
        torch.zeros(bs, 1),
        torch.ones(bs, 8),
        torch.full((bs, 1), 9)
    ], dim=1).long()  # shape (bs, max_seq_len+2)
    coord_target = torch.randint(
        low=0, high=608, size=(bs, 1+8, 4)
    )
    labels = {
        'html_target': html_target,
        'coord_target': coord_target,
    }
    print('html target shape: ', html_target.shape)
    print('coord target shape: ', coord_target.shape)
    # infer
    head.train()
    html_out, coord_out, html_hidden = head(fea, labels=None)
    print(html_out.shape, coord_out.shape, html_hidden.shape)
    pdb.set_trace()