import torch
import torch.nn as nn
import math

class LearnablePositisionEmbedding(nn.Module):
    def __init__(self, batch_size, fea_num, d_model):
        super(LearnablePositisionEmbedding, self).__init__()
        self.batch_size = batch_size
        self.fea_num = fea_num
        self.d_model = d_model
        # self.pos_embed = nn.Parameter(torch.zeros(self.batch_size, self.fea_num, self.d_model))

    def forward(self, x):
        b, f, e = x.shape
        self.pos_embed = nn.Parameter(torch.zeros(b, f, e)).cuda()
        # self.pos_embed = self.pos_embed.cuda()
        return x + self.pos_embed


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print(x.shape)
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.tokenConv(x).transpose(1, 2)
        # print(x.shape)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, c_in, fea_num, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4;
        hour_size = 24
        weekday_size = 7;
        day_size = 32;
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=fea_num,
                                   kernel_size=3, padding=padding, padding_mode='circular')

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        x = self.tokenConv(hour_x + weekday_x + day_x + month_x + minute_x)

        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        x = self.embed(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, batch_size, seq_len, pre_len, c_in, fea_num, d_model, flag, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        if flag == 'enc':
            self.input_len = seq_len
        else:
            self.input_len = pre_len
        self.value_embedding = TokenEmbedding(c_in=self.input_len, d_model=d_model)
        self.position_embedding = LearnablePositisionEmbedding(batch_size=batch_size, fea_num=fea_num, d_model=d_model)
        self.time_embedding = TemporalEmbedding(c_in=self.input_len, fea_num=fea_num, d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        x = self.position_embedding(x) + self.time_embedding(x_mark)
        return self.dropout(x)

