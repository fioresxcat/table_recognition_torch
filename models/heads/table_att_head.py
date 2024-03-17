from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .rec_att_head import AttentionGRUCell
from ..common_modules import *


class SLAHead(nn.Module):
    def __init__(self,
        list_in_c,
        hidden_size,
        out_c=30,
        max_text_length=500,
        loc_reg_num=4,
        fc_decay=0.0,
        **kwargs
    ):
        """
            @param in_channels: input shape
            @param hidden_size: hidden_size for RNN and Embedding
            @param out_channels: num_classes to rec
            @param max_text_length: max text pred
        """
        super().__init__()
        in_c = list_in_c[-1]
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_c
        self.loc_reg_num = loc_reg_num

        # structure
        self.structure_attention_cell = AttentionGRUCell(
            in_c, hidden_size, self.num_embeddings
        )
        self.structure_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, out_c)
        )
        # loc
        self.loc_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, loc_reg_num),
            nn.Sigmoid()   # loc is in [0, 1]
        )

    
    def _init_weights(self):
        """
            Uniform initialization
        """
        for module in [self.structure_generator, self.loc_generator]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    stdv = 1.0 / math.sqrt(self.hidden_size)
                    layer.weight.set_value(torch.uniform(-stdv, stdv))
                    layer.bias.set_value(torch.uniform(-stdv, stdv))


    def forward(self, inputs, labels=None):
            """
            Forward pass of the model.

            Args:
                inputs (list): List of input tensors (this is outputs of the neck module)
                targets (list, optional): List of label tensors (will have label during training). Defaults to None.

            Returns:
                dict: Dictionary containing the predicted structure probabilities and location predictions.
            """
            
            fea = inputs[-1]  # smallest, deepest feature return by neck
            batch_size = fea.shape[0]
            # reshape
            fea = rearrange(fea, 'b c h w -> b (h w) c')
            hidden = torch.zeros(batch_size, self.hidden_size, device=fea.device)
            structure_preds = torch.zeros(batch_size, self.max_text_length + 1, self.num_embeddings, device=fea.device)
            loc_preds = torch.zeros(batch_size, self.max_text_length + 1, self.loc_reg_num, device=fea.device)

            if self.training and labels is not None:
                structure = labels['structure']
                for i in range(self.max_text_length + 1):
                    hidden, structure_step, loc_step = self._decode(structure[:, i], fea, hidden)
                    structure_preds[:, i, :] = structure_step
                    loc_preds[:, i, :] = loc_step
            else:
                pre_chars = torch.zeros(batch_size, dtype=torch.int64, device=fea.device)
                # for export
                loc_step, structure_step = None, None
                for i in range(self.max_text_length + 1):
                    hidden, structure_step, loc_step = self._decode(pre_chars, fea, hidden)
                    pre_chars = structure_step.argmax(dim=1)
                    structure_preds[:, i, :] = structure_step
                    loc_preds[:, i, :] = loc_step

            return structure_preds, loc_preds


    def _decode(self, pre_chars, features, hidden):
        """
            Predict table label and coordinates for each step
            @param pre_chars: Table label in previous step
            @param features:
            @param hidden: hidden status in previous step
            @return:
        """
        emb_feature = self.emb(pre_chars)
        # output shape is b * self.hidden_size
        (output, hidden), alpha = self.structure_attention_cell(hidden, features, emb_feature)

        # structure
        structure_step = self.structure_generator(output)

        # loc
        loc_step = self.loc_generator(output)

        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        input_one_hot = F.one_hot(input_char, self.num_embeddings)
        return input_one_hot



if __name__ == '__main__':
    inputs = [
        torch.randn(2, 96, 56, 56),
        torch.randn(2, 96, 28, 28),
        torch.randn(2, 96, 14, 14)
    ]
    head = SLAHead(list_in_c=[96, 96, 96], hidden_size=256, out_c=30)
    out = head(inputs)
    print(out['structure_probs'].shape, out['loc_preds'].shape)
    pdb.set_trace()