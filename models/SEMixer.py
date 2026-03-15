__all__ = ['PatchTST_backbone']

# Cell
import time
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as f
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import scipy
import random
from layers.SelfAttention_Family import AttentionLayer,ProbAttention,LogSparseAttention,ReformerLayer,PerformerLayer
from sklearn.neighbors import kneighbors_graph
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from  layers.FourierCorrelation import FourierBlock
class InterPatchMixing(nn.Module):
    def __init__(self, PatchNum, connection_probability=0.1):
        super(InterPatchMixing, self).__init__()
        self.fc = nn.Linear(PatchNum, PatchNum)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=connection_probability)
        self.fc2 = nn.Linear(PatchNum, PatchNum)
        self.dropout2 = nn.Dropout(p=connection_probability)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class IntraPatchMixing(nn.Module):
    def __init__(self, PatchEmbedDim, connection_probability=0.1):
        super(IntraPatchMixing, self).__init__()
        self.fc1 = nn.Linear(PatchEmbedDim, PatchEmbedDim)
        self.dropout1 = nn.Dropout(p=connection_probability)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(PatchEmbedDim, PatchEmbedDim)
        self.dropout2 = nn.Dropout(p=connection_probability)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x




class TMBlock(nn.Module):
    def __init__(self, PatchNum, PatchEmbedDim, dropout, args=None):
        super(TMBlock, self).__init__()
        self.args = args
        self.inter_patch_mixing = InterPatchMixing(PatchNum, dropout)
        self.intra_patch_mixing = IntraPatchMixing(PatchEmbedDim, dropout)
        self.sample_num = args.sample_num
        self.PatchNum=PatchNum

        if self.args.prob_attn:
            self.self_attn = AttentionLayer(
                ProbAttention(False, args.factor, attention_dropout=args.dropout,
                              output_attention=args.output_attention),
                args.d_model, args.n_heads)
        elif self.args.logsparse:
            self.self_attn = AttentionLayer(
                LogSparseAttention(False, args.factor, attention_dropout=args.dropout,
                                   output_attention=args.output_attention),
                args.d_model, args.n_heads)
        elif self.args.peformer_attn:
            self.self_attn = AttentionLayer(
                PerformerLayer(None, args.d_model, args.n_heads),
                args.d_model, args.n_heads)
        elif self.args.reformer_attn: #locality-sensitive hashing (LSH)
            args.n_hashes = 4
            args.bucket_size = 4
            self.self_attn = AttentionLayer(
                ReformerLayer(None, args.d_model, args.n_heads, bucket_size=args.bucket_size,
                              n_hashes=args.n_hashes),
                args.d_model, args.n_heads)
        elif self.args.autocorrelation:
            self.self_attn = AutoCorrelationLayer(
                AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                output_attention=args.output_attention, args=args),
                args.d_model, args.n_heads, )
        elif self.args.fed_fourier_attn:
            self.self_attn = AutoCorrelationLayer(
                FourierBlock(in_channels=args.d_model,
                             out_channels=args.d_model,
                             seq_len=self.TimeNodesNum,
                             modes=args.modes,
                             mode_select_method=args.mode_select),
                args.d_model, args.n_heads, )
        elif args.self_attn:
            self.self_attn = _MultiheadAttention(args.d_model, args.n_heads, args.d_model, args.d_model,
                                                 attn_dropout=0,
                                                 proj_dropout=0, res_attention=False)
    
    
    def Random_Attention(self,X,connection_probability=None,mode='train'):
        effective_rate=1-connection_probability
        if mode=='train':
            mask = (torch.rand(X.shape) > connection_probability).float().to(X.device)
            return mask * X
        elif mode=='test':
            # Dropout ensemble
            return X * effective_rate


    def forward(self, x_or):
        res_x = x_or
        if self.args.Self_Attention_Mechanism:
            out, _ = self.self_attn(x_or)
            x_p = out.transpose(1, 2)
        elif self.args.Random_Attention_Mechanism:
            binary_mask_matrix=torch.full((self.PatchNum,self.PatchNum),1).to(x_or.device)
            if self.args.test:
                binary_mask_matrix=self.Random_Attention(binary_mask_matrix,connection_probability=self.args.connection_probability,mode='test')
            else:
                binary_mask_matrix=self.Random_Attention(binary_mask_matrix,connection_probability=self.args.connection_probability,mode='train')

            x_p = torch.matmul(binary_mask_matrix.unsqueeze(0), x_or)

            x_p = x_p.transpose(1, 2)
        else:
            x_p = x_or.transpose(1, 2)
        x_or = x_or.transpose(1, 2)
        x_in_f = self.inter_patch_mixing(x_p)
        x_p_e = x_in_f
        x = x_p_e.transpose(1, 2) + res_x
        res_x = x
        x = self.intra_patch_mixing(x) + res_x + x_or.transpose(1, 2)
        return x

class Model(nn.Module):
    def __init__(self, configs=None):

        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        head_dropout = configs.head_dropout
        individual = configs.individual
        affine = configs.affine
        subtract_last = configs.subtract_last
        # RevIn
        self.revin = True
        self.patch_len= configs.patch_len
        self.stride=configs.stride
        if self.revin: self.revin_layer = RevIN(self.configs.c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        padding_patch = self.configs.padding_patch
        self.scale_all = self.configs.scale_factors
        self.configs.scale_all = self.scale_all
        self.configs.patch_num_all = {}
        self.padding_patch_layer_all = {}
        for i in self.scale_all:
            self.configs.patch_num_all[i] = (int((self.configs.seq_len - self.configs.patch_len * i) / (self.configs.stride * i) + 1))
        if padding_patch == 'end':  # can be modified to general case
            for i, s in enumerate(self.scale_all):
                self.padding_patch_layer = nn.ReplicationPad1d((0, self.configs.stride * s))
                self.padding_patch_layer_all[s] = (self.padding_patch_layer)
                self.configs.patch_num_all[s] += 1

        if not self.configs.multi_scale:
            self.head_nf = d_model * self.configs.patch_num_all[1]
        else:
            self.head_nf = d_model * self.configs.reduce_dim
        # Backbone
        self.mpmc_layer = MPMCLayer(d_model=d_model,configs=self.configs)
        # Head
        self.n_vars = self.configs.c_in
        self.individual = individual
        self.head_all = []
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.configs.pred_len,
                                 head_dropout=head_dropout, args=configs)
        self.revin = True

    def forward(self, z):# z: [batch_size x sequence_length x variable_size]

        z = z.permute(0, 2, 1)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        z_scale_all = {}
        if self.configs.padding_patch == 'end':
            for i, s in enumerate(self.scale_all):
                z_scale_all[s] = (self.padding_patch_layer_all[s](z))
        for k, v in z_scale_all.items():
            z_scale_all[k] = v.unfold(dimension=-1, size=self.patch_len * k, step=self.stride * k).permute(0, 1, 3, 2)
        # z_scale_all: [batch_size x variable_size x patch_length x patch_number]

        z = self.mpmc_layer(z_scale_all)  # z: [batch_size x variable_size x d_model x patch_num]

        z = self.head(z)  # z: [batch_size x variable_size x prediction_length]
        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)

        return z.permute(0, 2, 1)
class MPMCLayer(nn.Module):
    def __init__(self,d_model,pe='zeros',learn_pe=True,configs=None):
        super().__init__()
        self.configs = configs
        self.W_Align_Scales = nn.ModuleList()
        self.W_Pos_Scales= []
        for s in self.configs.scale_all:
            self.W_Align_Scales.append(nn.Linear(self.configs.patch_len * s, self.configs.d_model))

        self.W_pos_1 = positional_encoding(pe, learn_pe, self.configs.patch_num_all[1], d_model)
        self.W_pos_2 = positional_encoding(pe, learn_pe, self.configs.patch_num_all[2], d_model)
        self.W_pos_3 = positional_encoding(pe, learn_pe, self.configs.patch_num_all[4], d_model)
        self.W_pos_4 = positional_encoding(pe, learn_pe, self.configs.patch_num_all[8], d_model)
        self.W_Pos_Scales= [self.W_pos_1, self.W_pos_2, self.W_pos_3, self.W_pos_4]

        self.T_Mixing_Scale_1 = nn.ModuleList()
        for i in range(self.configs.eib_num_1scale):
            self.T_Mixing_Scale_1.append(TMBlock(self.configs.patch_num_all[1], d_model, 0.1, args=configs))

        self.T_Mixing_Scale_2 = nn.ModuleList()
        for i in range(self.configs.eib_num):
            self.T_Mixing_Scale_2.append(
                TMBlock(self.configs.patch_num_all[1] + self.configs.patch_num_all[2], d_model, 0.1, args=configs))

        self.T_Mixing_Scale_3 = nn.ModuleList()
        for i in range(self.configs.eib_num):
            self.T_Mixing_Scale_3.append(
                TMBlock(self.configs.patch_num_all[2] + self.configs.patch_num_all[4], d_model, 0.1, args=configs))

        self.T_Mixing_Scale_4 = nn.ModuleList()
        for i in range(self.configs.eib_num):
            self.T_Mixing_Scale_4.append(
                TMBlock(self.configs.patch_num_all[4] + self.configs.patch_num_all[8], d_model, 0.1, args=configs))

        self.reduce = nn.Linear(sum(self.configs.patch_num_all.values()), self.configs.reduce_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x_scale_all_or) -> Tensor:  # [batch_size x variable_size x patch_length x patch_number]
        x_scale_all = {}
        n_vars = self.configs.enc_in
        for i, (k, v) in enumerate(x_scale_all_or.items()):
            x_scale_all[k] = self.W_Align_Scales[i](v.permute(0, 1, 3, 2))  # x: [batch_size x variable_size x patch_length x patch_embed_dim]

            x_temp = x_scale_all[k]
            x_scale_all[k] = torch.reshape(x_temp,(x_temp.shape[0] * x_temp.shape[1], x_temp.shape[2], x_temp.shape[3]))
        for i, (k, v) in enumerate(x_scale_all.items()):
            x_scale_all[k] = self.dropout(x_scale_all[k] + self.W_Pos_Scales[i])
        z_all = {}
        if not self.configs.multi_scale:
            for i in range(self.configs.eib_num):
                z_all[1] = self.T_Mixing_Scale_1[i](x_scale_all[1])
            z = z_all[1]
            z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
            z = z.permute(0, 1, 3, 2)
            return z
        else:
            z_final = {}
            for i in range(self.configs.eib_num_1scale):
                if i==0:
                    z_all[1] = self.T_Mixing_Scale_1[i](x_scale_all[1]) # [(batch_size x variable_size) x patch_number x patch_embed_dim]
                else:

                    z_all[1]= self.T_Mixing_Scale_1[i](z_all[1])
            temp = torch.cat([z_all[1], x_scale_all[2]], dim=1)

            for i in range(self.configs.eib_num):
                if i == 0:
                    z_all[2] = self.T_Mixing_Scale_2[i](temp)
                else:
                    z_all[2] = self.T_Mixing_Scale_2[i](z_all[2])
            temp = torch.cat([z_all[2][:, -x_scale_all[2].shape[1]:, :], x_scale_all[4]], dim=1)

            for i in range(self.configs.eib_num):
                if i == 0:
                    z_all[4] = self.T_Mixing_Scale_3[i](temp)
                else:
                    z_all[4] = self.T_Mixing_Scale_3[i](z_all[4])
            temp = torch.cat([z_all[4][:, -x_scale_all[4].shape[1]:, :], x_scale_all[8]], dim=1)
            for i in range(self.configs.eib_num):
                if i==0:
                    z_all[8] = self.T_Mixing_Scale_4[i](temp)
                else:
                    z_all[8]= self.T_Mixing_Scale_4[i](z_all[8])
            z_final[1] = z_all[1]
            z_final[2] = z_all[2][:, -x_scale_all[2].shape[1]:, :]
            z_final[4] = z_all[4][:, -x_scale_all[4].shape[1]:, :]
            z_final[8] = z_all[8][:, -x_scale_all[8].shape[1]:, :]
            z = torch.cat(list(z_final.values()), dim=-2)
            z = self.reduce(z.transpose(1, 2)).transpose(1, 2)
            z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
            z = z.permute(0, 1, 3, 2)

            return z

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, pred_len, head_dropout=0, args=None):
        super().__init__()
        self.args = args
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
                self.linears.append(nn.Linear(nf, pred_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        elif self.var_decomp:
            self.var_sp_num = args.var_sp_num  # 11

            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.var_sp_num):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, pred_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, pred_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x pred_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x pred_len]
        elif self.var_decomp:
            x_out = []
            output_chunks = torch.chunk(x, self.var_sp_num, dim=1)

            for i in range(len(output_chunks)):
                z = self.flattens[i](output_chunks[i])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x pred_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.cat(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)

        return x

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights