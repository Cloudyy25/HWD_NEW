from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d
import numpy as np

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        # **kwargs menangkap is_causal dari PyTorch >= 2.1
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = src.permute(1, 2, 0)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.permute(1, 2, 0)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)
        return src


# ─────────────────────────────────────────────────────────
# HWD ADDITION: CrossLevelAttention
# Dipakai oleh TSTransformerEncoder untuk menerima context
# dari level wavelet yang lebih kasar (approx / level atas).
# Q = representasi level ini, K/V = context dari level atas.
# ─────────────────────────────────────────────────────────
class CrossLevelAttention(nn.Module):
    """
    Cross-attention antara level wavelet.
    - x       : [seq_len, B, d_model]  — koefisien level ini
    - context : [seq_len, B, d_model]  — embedding dari level atas
    Output    : [seq_len, B, d_model]  — x yang sudah di-update (residual)
    """
    def __init__(self, d_model: int, nheads: int, dropout: float = 0.1):
        super(CrossLevelAttention, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=False   # [seq, B, d] agar konsisten dengan transformer lain
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # context di-interpolate ke panjang yang sama dengan x jika perlu
        if context.shape[0] != x.shape[0]:
            # [seq_ctx, B, d] → [B, d, seq_ctx] → interpolate → [B, d, seq_x] → [seq_x, B, d]
            ctx = context.permute(1, 2, 0)          # [B, d, seq_ctx]
            ctx = F.interpolate(ctx, size=x.shape[0], mode='linear', align_corners=False)
            ctx = ctx.permute(2, 0, 1)              # [seq_x, B, d]
        else:
            ctx = context

        attn_out, _ = self.attn(query=x, key=ctx, value=ctx)
        return self.norm(x + self.dropout(attn_out))  # residual + layer norm


# ─────────────────────────────────────────────────────────
# TSTransformerEncoder — dimodifikasi untuk HWD
# Perubahan dari FGTI asli:
#   1. __init__ : tambah self.cross_level_attn (CrossLevelAttention)
#   2. encoder  : terima context_emb=None; kalau ada, jalankan
#                 cross-attention sebelum self-attention transformer
# Semua method lain (masking, ssl) tidak berubah sama sekali.
# ─────────────────────────────────────────────────────────
class TSTransformerEncoder(nn.Module):

    def __init__(self, configs, dim_feedforward=128, dropout=0.1,
                 pos_encoding='learnable', activation='gelu', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.configs = configs
        self.max_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_heads = configs.nheads

        self.project_inp = nn.Linear(configs.enc_in * 2, self.d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            self.d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len
        )

        encoder_layer = TransformerBatchNormEncoderLayer(
            self.d_model, self.n_heads, dim_feedforward,
            dropout * (1.0 - freeze), activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, configs.e_layers)

        self.output_layer = nn.Linear(self.d_model, configs.enc_in)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = configs.enc_in
        self.criterion = nn.MSELoss(reduction="mean")

        # ── HWD: cross-level attention untuk menerima context dari level atas ──
        self.cross_level_attn = CrossLevelAttention(
            d_model=self.d_model,
            nheads=self.n_heads,
            dropout=dropout
        )

    def encoder(self, X, context_emb=None):
        """
        Args:
            X           : [B, seq_len, feat_dim]
            context_emb : [B, seq_len_ctx, d_model] atau None
                          Embedding dari level wavelet di atas (approx atau detail kasar).
                          Kalau None, perilaku identik dengan FGTI asli.
        Returns:
            output : [B, seq_len, d_model]
        """
        # [B, L, feat] → [L, B, feat] (konvensi PyTorch transformer)
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)  # [L, B, d_model]

        # ── HWD: kalau ada context dari level atas, lakukan cross-attention dulu ──
        if context_emb is not None:
            # context_emb masuk sebagai [B, L_ctx, d_model] → transpose ke [L_ctx, B, d_model]
            ctx = context_emb.permute(1, 0, 2)
            inp = self.cross_level_attn(inp, ctx)   # [L, B, d_model]

        output = self.transformer_encoder(inp)      # [L, B, d_model]
        output = self.act(output)
        output = output.permute(1, 0, 2)            # [B, L, d_model]
        return output

    # ── Semua method masking di bawah ini identik dengan FGTI asli ─────────────

    def get_random_mask(self, data_batch, mask_batch, mask_ratio_ssl, avg_mask_len):
        mask_ratio_ssl = mask_ratio_ssl + self.configs.missing_rate
        mask_temp = torch.rand_like(mask_batch)
        mask_temp[mask_temp <= mask_ratio_ssl] = 0
        mask_temp[mask_temp > mask_ratio_ssl] = 1
        mask_temp = mask_temp.to(self.configs.device)
        mask_ssl = mask_batch * mask_temp
        return mask_ssl

    def get_random_block_mask(self, data_batch, mask_batch, mask_ratio_ssl, avg_mask_len):
        mask_ratio_ssl = mask_ratio_ssl + self.configs.missing_rate
        mask_batch = mask_batch.detach().to("cpu").numpy()
        mask_ssl = []
        for sample_id in range(mask_batch.shape[0]):
            mask = mask_batch[sample_id]
            for j in range(mask.shape[1]):
                choice_num = int(mask.shape[0] * mask_ratio_ssl / avg_mask_len)
                mask_indexs = np.random.choice(mask.shape[0], choice_num)
                mask_lens = np.random.geometric(1 / avg_mask_len, choice_num)
                for i in range(choice_num):
                    mask[mask_indexs[i]: mask_indexs[i] + mask_lens[i], j] = 0
            mask_ssl.append(mask)
        mask_ssl = np.array(mask_ssl)
        mask_ssl = torch.from_numpy(mask_ssl).float().to(self.configs.device)
        return mask_ssl

    def get_tst_mask(self, data_batch, mask_batch, mask_ratio_ssl, avg_mask_len):
        mask_ratio_ssl = mask_ratio_ssl + self.configs.missing_rate
        mask_batch = mask_batch.detach().to("cpu").numpy()
        mask_ssl = []
        for sample_id in range(mask_batch.shape[0]):
            mask = mask_batch[sample_id]
            tst_mask = []
            for j in range(mask.shape[1]):
                keep_mask = np.ones(mask.shape[0])
                p_m = 1 / avg_mask_len
                p_u = p_m * mask_ratio_ssl / (1 - mask_ratio_ssl)
                p = [p_m, p_u]
                state = int(np.random.rand() > mask_ratio_ssl)
                for i in range(mask.shape[0]):
                    keep_mask[i] = state
                    if np.random.rand() < p[state]:
                        state = 1 - state
                tst_mask.append(keep_mask)
            tst_mask = np.vstack(tst_mask).T
            tst_mask = mask * tst_mask
            mask_ssl.append(tst_mask)
        mask_ssl = np.array(mask_ssl)
        mask_ssl = torch.from_numpy(mask_ssl).float().to(self.configs.device)
        return mask_ssl

    def self_supervised_learning(self, X, mask):
        ssl_label = X * mask
        mask_ssl = self.get_tst_mask(
            ssl_label, mask,
            mask_ratio_ssl=self.configs.mask_ratio_ssl,
            avg_mask_len=self.configs.avg_mask_len_ssl
        )
        X_input = X * mask_ssl
        output = self.encoder(X_input)   # context_emb=None → identik FGTI
        output = self.dropout1(output)
        output = self.output_layer(output)
        eval_ssl = mask - mask_ssl
        eval_p = torch.where(eval_ssl == 1)
        loss = self.criterion(output[eval_p], ssl_label[eval_p])
        return loss

    def forward(self, X):
        output = self.encoder(X)          # context_emb=None → identik FGTI
        output = self.dropout1(output)
        output = self.output_layer(output)
        return output