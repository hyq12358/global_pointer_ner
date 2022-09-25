#!/usr/bin/env python3

import torch
from torch import nn
from transformers.models.bert import BertModel, BertConfig


class GlobalPointer(nn.Module):
    def __init__(self, ent_type_size=10, RoPE=True):
        super().__init__()
        self.config = BertConfig.from_pretrained("./roberta_pretrain/config.json")
        self.encoder = BertModel.from_pretrained(
            "./roberta_pretrain", config=self.config
        )
        self.ent_type_size = ent_type_size
        self.inner_dim = 64
        self.hidden_size = self.config.hidden_size
        self.dense = nn.Linear(
            self.hidden_size, self.ent_type_size * self.inner_dim * 2
        )
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(
            0, seq_len, dtype=torch.float, device=device
        ).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        device = input_ids.device
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]  # (batch_size, max_len, hidden_size)
        batch_size, seq_len, hidden_size = last_hidden_state.size()

        outputs = self.dense(
            last_hidden_state
        )  # (batch_size, max_len, label_num*inner_dim*2)

        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # 把outputs按最后一个维度按照self.inner_dim*2的size进行切分，每个的维度是(batch_size, max_len, self.inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # stack之后，outputs的维度是(batch_size, max_len, self.label_num, self.inner_dim*2)

        qw, kw = outputs[..., : self.inner_dim], outputs[..., self.inner_dim :]
        # qw, kw: batch_size, max_len, label_num, inner_dim
        # q[:,i,k,:]*k[:,j,k,:] which represent the score label[k] start from index i to index j
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(
                batch_size, seq_len, self.inner_dim, device
            )
            # pose_emb: batch_size, max_len, inner_dim
            # cos_pos: batch_size, max_en, 1, inner_dim
            # sin_pos: batch_size, max_en, 1, inner_dim
            # https://spaces.ac.cn/archives/8265
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos

            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # b: batch_size
        # m: max_len
        # h: label_num
        # d: inner_dim
        logits = torch.einsum(
            "bmhd, bnhd->bhmn", qw, kw
        )  # batch_size, label_num, max_len, max_len
        # attention_mask: batch_size, max_len
        pad_mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self.ent_type_size, seq_len, seq_len)
        )
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits / self.inner_dim**0.5  # batch_size, label_num, max_len, max_len
