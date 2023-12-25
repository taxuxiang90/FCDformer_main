import torch
import torch.nn as nn
import torch.nn.functional as F

from model.enc import EncoderLayer, Convlayer, Fre_EncoderLayer
from model.dec import DecoderLayer, Convlayer
from model.embed import DataEmbedding
from model.cross_atten import Attenlayer, Fre_Attenlayer



class FIformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, pre_len, fea_num,
                 factor=5, d_model=512, batch_size=32, n_heads=8, e_layers=3, d_layers=1, d_ff=32,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(FIformer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.e_layers = e_layers
        self.d_layers = d_layers
        # Encoding
        self.enc_embedding = DataEmbedding(batch_size=batch_size, seq_len=seq_len, pre_len=pre_len, c_in=enc_in,
                                           fea_num=fea_num, d_model=d_model, flag='enc', embed_type=embed, freq=freq, dropout=dropout)
        self.encoder = EncoderLayer(d_model=d_model, n_heads=n_heads, fea_num=fea_num, batch_size=batch_size, d_ff=d_ff)
        self.dec_embedding = DataEmbedding(batch_size=batch_size, seq_len=seq_len, pre_len=pre_len, c_in=enc_in,
                                           fea_num=fea_num, d_model=d_model, flag='dec', embed_type=embed, freq=freq, dropout=dropout)
        self.decoder = DecoderLayer(d_model=d_model, n_heads=n_heads, fea_num=fea_num, batch_size=batch_size, d_ff=d_ff)
        self.cross_atten = Attenlayer(d_model=d_model, n_heads=n_heads, fea_num=fea_num, batch_size=batch_size, d_ff=d_ff)
        self.tokenConv = nn.Conv1d(in_channels=fea_num, out_channels=pre_len,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, batch_x, batch_y, x_time, y_time,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(batch_x, x_time)
        for l in range(self.e_layers):
            enc_out, k_layer = self.encoder(enc_out)
            self.enc_out = enc_out
            self.k_layer = k_layer
        dec_out = self.dec_embedding(batch_y, y_time)
        for l in range(self.d_layers):
            dec_out, q_layer = self.decoder(dec_out)
            self.dec_out = dec_out
            self.q_layer = q_layer
        out = self.cross_atten(dec_out, self.q_layer, self.k_layer)
        out = self.tokenConv(out)
        out = self.projection(out)

        return out


class FCDformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, pre_len, fea_num,
                 factor=5, d_model=512, batch_size=32, fd_num=2, n_heads=8, e_layers=3, d_layers=1, d_ff=32,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(FCDformer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.e_layers = e_layers
        self.d_layers = d_layers
        # Encoding
        self.enc_embedding = DataEmbedding(batch_size=batch_size, seq_len=seq_len, pre_len=pre_len, c_in=enc_in,
                                           fea_num=fea_num, d_model=d_model, flag='enc', embed_type=embed, freq=freq, dropout=dropout)
        self.encoder = Fre_EncoderLayer(d_model=d_model, n_heads=n_heads, fea_num=fea_num, batch_size=batch_size, d_ff=d_ff, fd_num=fd_num)
        self.dec_embedding = DataEmbedding(batch_size=batch_size, seq_len=seq_len, pre_len=pre_len, c_in=enc_in,
                                           fea_num=fea_num, d_model=d_model, flag='dec', embed_type=embed, freq=freq, dropout=dropout)
        self.decoder = DecoderLayer(d_model=d_model, n_heads=n_heads, fea_num=fea_num, batch_size=batch_size, d_ff=d_ff)
        self.cross_atten = Fre_Attenlayer(d_model=d_model, n_heads=n_heads, fea_num=fea_num, batch_size=batch_size, d_ff=d_ff)
        self.tokenConv = nn.Conv1d(in_channels=fea_num, out_channels=pre_len,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, batch_x, batch_y, x_time, y_time,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(batch_x, x_time)
        for l in range(self.e_layers):
            enc_out = self.encoder(enc_out)
            self.enc_out = enc_out
        dec_out = self.dec_embedding(batch_y, y_time)
        for l in range(self.d_layers):
            dec_out = self.decoder(dec_out)
            self.dec_out = dec_out
        out = self.cross_atten(self.dec_out, self.enc_out)
        out = self.tokenConv(out)
        out = self.projection(out)

        return out


