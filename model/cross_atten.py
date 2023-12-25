import torch
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import math

class Attenlayer(nn.Module):
    def __init__(self, d_model, n_heads, fea_num, batch_size, d_ff):
        super(Attenlayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.batch_size = batch_size
        self.fea_num = fea_num
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num = int(self.d_model / self.n_heads)
        self.softmax = Softmax(dim=-1)
        self.conv = Convlayer(d_model=d_model, d_ff=d_ff)

    def forward(self, x, query_layer, key_layer):
        b, f, e = x.shape

        query_layer = query_layer.reshape(b, self.n_heads, f, -1)
        key_layer = key_layer.reshape(b, self.n_heads, f, -1)

        value_layer = x.reshape(b, self.n_heads, f, -1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.n_heads)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.reshape(b, f, -1)

        context_layer = self.conv(context_layer)
        context_layer = context_layer + x

        return context_layer


class Fre_Attenlayer(nn.Module):
    def __init__(self, d_model, n_heads, fea_num, batch_size, d_ff):
        super(Fre_Attenlayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.batch_size = batch_size
        self.fea_num = fea_num
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num = int(self.d_model / self.n_heads)
        self.softmax = Softmax(dim=-1)
        self.conv = Convlayer(d_model=d_model, d_ff=d_ff)
        self.unsampleconv = nn.Conv1d(in_channels=int(self.fea_num * 2), out_channels=self.fea_num, kernel_size=3,
                                      padding=padding, padding_mode='circular')
        self.Query = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.Key = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.Value = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')


    def forward(self, dec_x, enc_x):
        b, f, e = dec_x.shape
        # emb_len = int(e / self.n_heads)
        x_dec_query_layer = self.Query(dec_x.permute(0, 2, 1)).transpose(1, 2)
        x_dec_key_layer = self.Key(dec_x.permute(0, 2, 1)).transpose(1, 2)
        x_dec_value_layer = self.Value(dec_x.permute(0, 2, 1)).transpose(1, 2)

        x_enc_query_layer = self.Query(enc_x.permute(0, 2, 1)).transpose(1, 2)
        x_enc_key_layer = self.Key(enc_x.permute(0, 2, 1)).transpose(1, 2)
        x_enc_value_layer = self.Value(enc_x.permute(0, 2, 1)).transpose(1, 2)

        x_dec_query_layer = x_dec_query_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
        x_dec_key_layer = x_dec_key_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
        x_dec_value_layer = x_dec_value_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)

        x_enc_query_layer = x_enc_query_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
        x_enc_key_layer = x_enc_key_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
        x_enc_value_layer = x_enc_value_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)

        dec_attention_scores = torch.matmul(x_dec_query_layer, x_enc_key_layer.transpose(-1, -2))
        dec_attention_scores = dec_attention_scores / math.sqrt(self.n_heads)
        dec_attention_probs = self.softmax(dec_attention_scores)
        dec_context_layer = torch.matmul(dec_attention_probs, x_enc_value_layer)
        dec_context_layer = dec_context_layer.permute(0, 2, 1, 3).contiguous()
        dec_context_layer = dec_context_layer.reshape(b, f, -1)

        enc_attention_scores = torch.matmul(x_enc_query_layer, x_dec_key_layer.transpose(-1, -2))
        enc_attention_scores = enc_attention_scores / math.sqrt(self.n_heads)
        enc_attention_probs = self.softmax(enc_attention_scores)
        enc_context_layer = torch.matmul(enc_attention_probs, x_dec_value_layer)
        enc_context_layer = enc_context_layer.permute(0, 2, 1, 3).contiguous()
        enc_context_layer = enc_context_layer.reshape(b, f, -1)

        context_layer = torch.cat((enc_context_layer, dec_context_layer), dim=1)

        context_layer = self.unsampleconv(context_layer)

        context_layer = context_layer + dec_x

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