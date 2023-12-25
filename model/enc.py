import torch
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
import math
import pywt


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, fea_num, batch_size, d_ff):
        super(EncoderLayer, self).__init__()
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
        self.conv = Convlayer(d_model=self.d_model, d_ff=self.d_ff)
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


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.n_heads)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        b, f, h, e = context_layer.shape
        context_layer = context_layer.reshape(b, f, h * e)

        context_layer = self.conv(context_layer)

        context_layer = context_layer + x

        return context_layer, key_layer

class Fre_EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, fea_num, batch_size, d_ff, fd_num):
        super(Fre_EncoderLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.batch_size = batch_size
        self.fea_num = fea_num
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.fd_num = fd_num
        self.num = int(self.d_model / self.n_heads)
        self.Query = nn.Conv1d(in_channels=int(d_model / 2 ** fd_num), out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.Key = nn.Conv1d(in_channels=int(d_model / 2 ** fd_num), out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.Value = nn.Conv1d(in_channels=int(d_model / 2 ** fd_num), out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.conv = Convlayer(d_model=self.d_model, d_ff=self.d_ff, fd_num=fd_num)
        self.upsizeconv = upsizeConvlayer(d_model=self.d_model, d_ff=self.d_ff, fd_num=fd_num)
        self.softmax = Softmax(dim=-1)
        self.unsampleconv = nn.Conv1d(in_channels=int(self.fea_num * 2), out_channels=self.fea_num, kernel_size=3, padding=padding, padding_mode='circular')

    def forward(self, x):
        global context_layer
        x_input = x
        x = x.cpu().detach().numpy()
        for i in range(self.fd_num):
            x = pywt.dwt(x, 'db1', axis=-1)

        x1, x2 = x[0], x[1]
        x1 = torch.FloatTensor(x1).cuda()
        x2 = torch.FloatTensor(x2).cuda()

        B, F, D = x1.shape[-3], x1.shape[-2], x1.shape[-1]

        x1 = x1.reshape(-1, B, F, D)
        x2 = x2.reshape(-1, B, F, D)

        x = torch.cat((x1, x2), dim=0)
        N = x.shape[0]

        x_fre1 = x[0:1, :, :, :]
        x_fre1 = x_fre1.squeeze(dim=0)
        for i in range(N - 1):
            x_fre2 = x[i+1:i+2, :, :, :]
            x_fre2 = x_fre2.squeeze(dim=0)
            x_h_query_layer = self.Query(x_fre1.permute(0, 2, 1)).transpose(1, 2)
            x_h_key_layer = self.Key(x_fre1.permute(0, 2, 1)).transpose(1, 2)
            x_h_value_layer = self.Value(x_fre1.permute(0, 2, 1)).transpose(1, 2)

            x_l_query_layer = self.Query(x_fre2.permute(0, 2, 1)).transpose(1, 2)
            x_l_key_layer = self.Key(x_fre2.permute(0, 2, 1)).transpose(1, 2)
            x_l_value_layer = self.Value(x_fre2.permute(0, 2, 1)).transpose(1, 2)

            b, f, _ = x_fre2.shape

            x_h_query_layer = x_h_query_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
            x_h_key_layer = x_h_key_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
            x_h_value_layer = x_h_value_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)

            x_l_query_layer = x_l_query_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
            x_l_key_layer = x_l_key_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)
            x_l_value_layer = x_l_value_layer.reshape(b, f, self.n_heads, -1).permute(0, 2, 1, 3)


            h_attention_scores = torch.matmul(x_h_query_layer, x_l_key_layer.transpose(-1, -2))
            h_attention_scores = h_attention_scores / math.sqrt(self.n_heads)
            h_attention_probs = self.softmax(h_attention_scores)
            h_context_layer = torch.matmul(h_attention_probs, x_h_value_layer)
            h_context_layer = h_context_layer.permute(0, 2, 1, 3).contiguous()
            h_context_layer = h_context_layer.reshape(b, f, -1)

            l_attention_scores = torch.matmul(x_l_query_layer, x_h_key_layer.transpose(-1, -2))
            l_attention_scores = l_attention_scores / math.sqrt(self.n_heads)
            l_attention_probs = self.softmax(l_attention_scores)
            l_context_layer = torch.matmul(l_attention_probs, x_l_value_layer)
            l_context_layer = l_context_layer.permute(0, 2, 1, 3).contiguous()
            l_context_layer = l_context_layer.reshape(b, f, -1)

            context_layer = torch.cat((h_context_layer, l_context_layer), dim=1)
            context_layer = self.unsampleconv(context_layer)

            context_layer = self.conv(context_layer)
            x_fre1 = context_layer
        context_layer = self.upsizeconv(context_layer)
        context_layer = context_layer + x_input
        return context_layer


class Convlayer(nn.Module):
    def __init__(self, d_model, d_ff, fd_num):
        super(Convlayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fd_num = fd_num
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv_token = nn.Sequential(
                                nn.Conv1d(in_channels=self.d_model, out_channels=self.d_ff,
                                    kernel_size=3, padding=padding, padding_mode='circular'),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=self.d_ff, out_channels=int(self.d_model / (2 ** self.fd_num)),
                                    kernel_size=3, padding=padding, padding_mode='circular')
        )
        self.norm_layer = torch.nn.LayerNorm(int(self.d_model / (2 ** self.fd_num)))

    def forward(self, x):
        x = self.conv_token(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.norm_layer(x)
        return x

class upsizeConvlayer(nn.Module):
    def __init__(self, d_model, d_ff, fd_num):
        super(upsizeConvlayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fd_num = fd_num
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv_token = nn.Sequential(
                                nn.Conv1d(in_channels=int(self.d_model / (2 ** self.fd_num)), out_channels=self.d_ff,
                                    kernel_size=3, padding=padding, padding_mode='circular'),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=self.d_ff, out_channels=self.d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        )
        self.norm_layer = torch.nn.LayerNorm(self.d_model)

    def forward(self, x):
        x = self.conv_token(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.norm_layer(x)
        return x
