import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed_Itran import DataEmbedding_inverted
import numpy as np

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0, args=None):
        super().__init__()
        self.args = args
        # self.individual = individual
        self.individual = args.var_individual
        self.n_vars = n_vars
        self.sp_patch_num = 4
        self.var_decomp = args.var_decomp
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        elif args.patch_decomp:
            self.linears = nn.ModuleList()
            self.flatten = nn.Flatten(start_dim=-2)
            for _ in range(self.sp_patch_num):
                self.linears.append(nn.Linear(16 * 16, target_window))
            self.linears.append(nn.Linear(nf, target_window))
        elif self.var_decomp:
            self.var_sp_num = args.var_sp_num  # 11
            # print('var_sp_num', self.var_sp_num)
            # time.sleep(500)
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.var_sp_num):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # print('1111')
            # time.sleep(500)
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            if args.linear_mlp:
                self.linear = nn.Sequential(nn.Linear(nf, nf // 2), nn.Linear(nf // 2, target_window))
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
                # print('1111')
                # time.sleep(500)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        elif self.args.patch_decomp:  # 变量多的话分批预测
            x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2], -1, self.sp_patch_num))
            # print(x.shape)
            f = 0
            for i in range(self.sp_patch_num):
                z = self.flatten(x[:, :, :, :, i])
                z = self.linears[i](z)
                f += z
            z = self.flatten(x)
            z = self.flatten(z)
            z = self.linears[i + 1](z)
            f += z
            return f
        elif self.var_decomp:
            x_out = []
            # print(x.shape)
            # print(x.shape)

            output_chunks = torch.chunk(x, self.var_sp_num, dim=1)

            for i in range(len(output_chunks)):
                z = output_chunks[i]  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.cat(x_out, dim=1)
            # print(x.shape)
            # time.sleep(500)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)

        return x
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs=configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # self.projector2 = Flatten_Head(False, self.configs.c_in, 512, configs.pred_len,
        #                          head_dropout=0, args=configs)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # if self.use_norm:
        #     x=self.projector2(enc_out[:,:N,:])
        #     print(enc_out.shape,x_enc.shape,self.use_norm)
        #     print(self.projector(enc_out).shape,self.configs.d_model)
        #     time.sleep(500)
        # B N E -> B N S -> B S N
        # dec_out = self.projector2(enc_out[:, :N, :]).permute(0, 2, 1)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]