import copy
import math
from typing import Union, Callable, Optional, Tuple, Any

import copy
from torch import nn, Tensor
import torch.nn.functional as F

import torch
import random

from torch import nn, Tensor

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]
        #return output + residual


class Transformer(nn.Module):

    def __init__(self, nvoc, ninput, nhead, nhid, nlayers, dropout=0.5):
        super(Transformer, self).__init__()
        self.ninput = ninput
        self.encoder = nn.Embedding(nvoc, ninput)
        self.pos_encoder = PositionalEncoding(ninput, dropout)

        encoder_layers = TransformerEncoderLayer(ninput, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninput, nvoc)

        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, mask):
        embeddings = self.encoder(input) * math.sqrt(self.ninput)
        pos_embeddings = self.pos_encoder(embeddings)
        output, attn_output_weights = self.transformer_encoder(pos_embeddings, mask)
        decoded = self.decoder(output)
        return decoded, attn_output_weights

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> \
            Tuple[Union[Tensor, Any], Any]:
        output = src
        for mod in self.layers:
            output, attn_output_weights = mod(output, attn_mask=mask, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_output_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, ninput, nhead, nhid, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.pos_ffn = PoswiseFeedForwardNet(ninput, nhid)
        self.enc_self_attn = nn.MultiheadAttention(ninput, nhead, dropout)

    def forward(self, enc_inputs, attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tuple:
        #enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               attn_mask=attn_mask,
                                               key_padding_mask=key_padding_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        #enc_outputs = self.pos_ffn(enc_inputs)
        #attn = None
        return enc_outputs, attn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
