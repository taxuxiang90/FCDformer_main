import torch
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import math


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, fea_num, batch_size, d_ff):
        super(DecoderLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.batch_size = batch_size
        self.fea_num = fea_num
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num = int(self.d_model / self.n_heads)
        self.Query = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.Key = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.Value = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.conv = Convlayer(d_model=d_model, d_ff=d_ff)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        b, f, e = x.shape
        emb_len = int(e / self.n_heads)
        query_layer = self.Query(x.permute(0, 2, 1)).transpose(1, 2)
        key_layer = self.Key(x.permute(0, 2, 1)).transpose(1, 2)
        value_layer = self.Value(x.permute(0, 2, 1)).transpose(1, 2)

        query_layer = query_layer.reshape(b, f, self.n_heads, emb_len).permute(0, 2, 1, 3)
        key_layer = key_layer.reshape(b, f, self.n_heads, emb_len).permute(0, 2, 1, 3)
        value_layer = value_layer.reshape(b, f, self.n_heads, emb_len).permute(0, 2, 1, 3)

        # query_layer = query_layer.permute(0, 2, 1, 3)
        # key_layer = key_layer.permute(0, 2, 1, 3)
        # value_layer = value_layer.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.n_heads)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        b, f, h, e = context_layer.shape
        context_layer = context_layer.reshape(b, f, h * e)

        context_layer = self.conv(context_layer)
        context_layer = context_layer + x

        return context_layer


class Convlayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(Convlayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv_token = nn.Sequential(
                                nn.Conv1d(in_channels=self.d_model, out_channels=self.d_ff,
                                    kernel_size=3, padding=padding, padding_mode='circular'),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=self.d_ff, out_channels=self.d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
                                )
        self.norm_layer = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv_token(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.norm_layer(x)
        return x