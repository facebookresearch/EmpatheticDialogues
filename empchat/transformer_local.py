# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from empchat.datasets.tokens import PAD_TOKEN


def create_position_codes(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class TransformerModel(nn.Module):
    def __init__(
        self,
        transformer_n_heads,
        transformer_n_layers,
        transformer_dim,
        vocabulary_dim,
        embedding=None,
        fix_mean=True,
        dropout=0,
        padding_idx=None,
    ):
        super(TransformerModel, self).__init__()
        self.n_layers = transformer_n_layers
        self.fix_mean = fix_mean
        n_heads = transformer_n_heads  # 8 by default
        dim = transformer_dim  # 512 by default
        self.out_dim = dim
        dim_hidden = dim * 4  # 2048 by default
        assert dim % n_heads == 0, "transformer dim must be a multiple of n_heads"
        n_positions = 1000
        self.position_embeddings = nn.Embedding(n_positions, dim)
        create_position_codes(n_positions, dim, out=self.position_embeddings.weight)
        if embedding is not None:
            self.embeddings = embedding
        elif padding_idx is not None:
            self.embeddings = nn.Embedding(vocabulary_dim, dim, padding_idx=padding_idx)
        else:
            self.embeddings = nn.Embedding(vocabulary_dim, dim)
        self.dim = dim
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(n_heads, dim, dropout=dropout))
            self.layer_norm1.append(nn.LayerNorm([dim]))
            self.ffns.append(TransformerFFN(dim, dim_hidden, dropout=dropout))
            self.layer_norm2.append(nn.LayerNorm([dim]))

    def forward(self, input_, mask):
        """
        input data is a LongTensor of shape [batch, seq_len], containing each
        word's index in the embeddings table.
        mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
        inside the sequence and 0 outside.
        """
        seq_len = input_.size(1)
        positions = input_.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input_)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.n_layers):
            tensor = tensor + self.attentions[i](tensor, mask)
            tensor = self.normalize(tensor, self.layer_norm1[i])
            tensor = tensor + self.ffns[i](tensor, mask)
            tensor = self.normalize(tensor, self.layer_norm2[i])
            tensor *= mask.unsqueeze(-1).float()
        if self.fix_mean:
            output = tensor.sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)
        else:
            output = tensor.mean(dim=1)
        return output

    def normalize(self, tensor, norm_layer):
        size = tensor.size()
        return norm_layer(tensor.view(-1, self.dim)).view(size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        # multi head is seen as one layer, dropout is only applied to the input
        self.in_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, input_, mask):
        # Input is [B, seq_len, dim]
        # Mask is [B, seq_len]
        batch_size, seq_len, dim = input_.size()
        assert (
            dim == self.dim
        ), f"Dimensions do not match: {dim} input vs {self.dim} configured"
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            tensor = tensor.view(batch_size, seq_len, n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        in_droped = self.in_dropout(input_)
        query = prepare_head(self.q_lin(in_droped))
        keys = prepare_head(self.k_lin(in_droped))
        values = prepare_head(self.v_lin(in_droped))
        scale = math.sqrt(dim_per_head)
        dot_prod = query.bmm(keys.transpose(1, 2))
        # [B * n_heads, seq_len, seq_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, 1, seq_len)
            .repeat(1, n_heads, seq_len, 1)
            .view(batch_size * n_heads, seq_len, seq_len)
        )
        dot_prod.masked_fill_(attn_mask, -float("inf"))
        attn_weights = F.softmax(dot_prod / scale, dim=-1)
        attentioned = attn_weights.bmm(values)
        attentioned = (
            attentioned.view(batch_size, n_heads, seq_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, dim)
        )
        return self.out_lin(attentioned)


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, dropout=0):
        super(TransformerFFN, self).__init__()
        self.in_dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)

    def forward(self, input_, mask):
        return self.lin2(F.relu(self.lin1(self.in_dropout(input_))))


class TransformerAdapter(nn.Module):
    def __init__(self, opt, dictionary):
        super(TransformerAdapter, self).__init__()
        self.opt = opt
        self.pad_idx = dictionary[PAD_TOKEN]
        self.embeddings = nn.Embedding(
            len(dictionary), opt.embeddings_size, padding_idx=self.pad_idx
        )
        if not opt.learn_embeddings:
            self.embeddings.weight.requires_grad = False
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.05)
        dropout = opt.transformer_dropout if opt.transformer_dropout else 0
        self.ctx_transformer = TransformerModel(
            opt.transformer_n_heads,
            opt.n_layers,
            opt.transformer_dim,
            len(dictionary),
            embedding=self.embeddings,
            dropout=dropout,
        )
        self.cand_transformer = TransformerModel(
            opt.transformer_n_heads,
            opt.n_layers,
            opt.transformer_dim,
            len(dictionary),
            embedding=self.embeddings,
            dropout=dropout,
        )
        self.embeddings = self.ctx_transformer.embeddings

    def forward(self, context_w, cands_w):
        if context_w is not None:
            context_mask = context_w != self.pad_idx
            context_h = self.ctx_transformer(context_w, context_mask)
            if self.opt.normalize_sent_emb:
                context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
        else:
            context_h = None
        if cands_w is not None:
            cands_mask = cands_w != self.pad_idx
            cands_h = self.cand_transformer(cands_w, cands_mask)
            if self.opt.normalize_sent_emb:
                cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)
        else:
            cands_h = None
        return context_h, cands_h
