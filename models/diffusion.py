"""
diffusion.py — HWD
Base: FGTI models/diffusion.py

Perubahan dari FGTI:
  1. Import: hapus torch.fft, ganti dengan pywt
  2. NoiseProject : KEEP identik FGTI (class ini ada di sini, bukan di Diff_layers)
  3. DiffusionEmbedding : KEEP identik FGTI
  4. Conv1d_with_init   : KEEP identik FGTI
  5. diff_HWD (ganti diff_FGTI):
     - Hapus FFT high_freq / dominant_freq
     - Ganti satu TSTransformerEncoder flat dengan:
         approx_encoder (lebih dalam) + detail_encoders (satu per level)
     - _build_conditions: sequential approx→detail via CrossLevelAttention
     - forward(): identik FGTI kecuali kondisi datang dari wavelet
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

from layers.Diff_layers import TemporalLearning, FeatureLearning
from models.ts_transformer import TSTransformerEncoder


# ── Identik FGTI ────────────────────────────────────────────────────

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps       = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (
            torch.arange(dim) / (dim - 1) * 4.0
        ).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class NoiseProject(nn.Module):
    """Identik FGTI — tidak ada perubahan sama sekali."""
    def __init__(self, side_dim, channels, diffusion_embedding_dim,
                 nheads, target_dim, proj_t,
                 is_cross_t=True, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection      = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection       = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection    = Conv1d_with_init(channels, 2 * channels, 1)
        self.forward_time         = TemporalLearning(
            channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature      = FeatureLearning(
            channels=channels, nheads=nheads,
            target_dim=target_dim, proj_t=proj_t, is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, guide_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, guide_info)
        y = self.forward_feature(y, base_shape, guide_info)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)
        y = y + side_info

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x        = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip     = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# ── HWD wavelet utilities ────────────────────────────────────────────

def _dwt(x_np: np.ndarray, wavelet: str, level: int):
    """
    x_np : [B, K, L] float32
    return: [cA_L, cD_L, ..., cD_1]  (approx dulu, detail kasar→halus)
    Level di-clamp otomatis agar tidak melebihi batas pywt untuk seq length L.
    """
    L_seq   = x_np.shape[-1]
    max_lvl = pywt.dwt_max_level(L_seq, wavelet)
    level   = min(level, max(1, max_lvl - 1))   # aman: 1 level di bawah max
    raw     = pywt.wavedec(x_np, wavelet=wavelet, level=level, axis=-1)
    approx  = raw[0]
    details = list(reversed(raw[1:]))   # [kasar, ..., halus]
    return [approx] + details


def _idwt(coeffs_np: list, wavelet: str) -> np.ndarray:
    approx  = coeffs_np[0]
    details = list(reversed(coeffs_np[1:]))
    return pywt.waverec([approx] + details, wavelet=wavelet, axis=-1)


def _resize_mask(mask_np: np.ndarray, target_len: int) -> np.ndarray:
    """Downscale mask [B, K, L] ke [B, K, target_len] via avg pool."""
    B, K, L = mask_np.shape
    t = torch.tensor(mask_np, dtype=torch.float32)
    t = t.reshape(B * K, 1, L)
    t = F.adaptive_avg_pool1d(t, output_size=target_len)
    return (t.reshape(B, K, target_len).numpy() > 0.5).astype(np.float32)


# ── diff_HWD ─────────────────────────────────────────────────────────

class diff_HWD(nn.Module):
    """
    Menggantikan diff_FGTI.

    Perbedaan utama:
      - Satu TSTransformerEncoder per level wavelet (approx + detail × L)
      - Conditioning dibangun secara sequential:
          approx dulu → context → detail kasar → ... → detail halus
      - Masking: observed coefficients jadi anchor, missing = noise init
      - NoiseProject, DiffusionEmbedding, input/output projection: IDENTIK FGTI
    """

    def __init__(self, configs):
        super().__init__()
        self.configs     = configs
        self.channel     = configs.channel
        self.wavelet     = getattr(configs, 'wavelet', 'db4')
        self.wav_levels  = getattr(configs, 'levels', 3)

        # Auto-clamp levels berdasarkan seq_len agar aman saat DWT
        import pywt as _pywt
        _req     = self.wav_levels
        max_safe = max(1, _pywt.dwt_max_level(configs.seq_len, self.wavelet) - 1)
        self.wav_levels = min(self.wav_levels, max_safe)
        if self.wav_levels < _req:
            print(f'[diff_HWD] PERINGATAN: levels diminta {_req}, '
                  f'dipangkas jadi {self.wav_levels} '
                  f'(wavelet={self.wavelet}, seq_len={configs.seq_len})')
        else:
            print(f'[diff_HWD] levels aktif: {self.wav_levels} '
                  f'(wavelet={self.wavelet})')
        
        # Flag arsitektur v2 (tiga perbaikan penyaluran sinyal)
        self.arch_v2 = getattr(configs, 'arch_v2', False)

        n_branches       = self.wav_levels + 1   # approx + detail × L

        # ── Satu encoder per level (approx lebih dalam) ──────────────
        approx_cfg          = _deeper(configs, factor=2)
        self.approx_encoder = TSTransformerEncoder(approx_cfg)
        self.detail_encoders = nn.ModuleList([
            TSTransformerEncoder(configs) for _ in range(self.wav_levels)
        ])

        # ── Projection layers — pola identik FGTI ────────────────────
        # Project d_model → channel langsung — memory efficient
        self.branch_projections = nn.ModuleList([
            Conv1d_with_init(configs.d_model, self.channel, 1)
            for _ in range(n_branches)
        ])
        # PERBAIKAN 3: penggabungan berbobot, bukan rata-rata seragam
        self.branch_merge = Conv1d_with_init(
            self.channel * n_branches, self.channel, 1)
        self.side_projection       = Conv1d_with_init(configs.side_dim, self.channel, 1)
        self.condition_projection2 = Conv1d_with_init(self.channel, self.channel, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=configs.diffusion_step_num,
            embedding_dim=configs.d_model
        )

        # PERBAIKAN 1: kanal input 2 → 1 bila arch_v2 aktif
        in_ch = 1 if self.arch_v2 else 2
        self.input_projection    = Conv1d_with_init(in_ch, self.channel, 1)
        self.output_projection1  = Conv1d_with_init(self.channel, self.channel, 1)
        self.output_projection2  = Conv1d_with_init(self.channel, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList([
            NoiseProject(
                side_dim=configs.side_dim,
                channels=self.channel,
                diffusion_embedding_dim=configs.d_model,
                nheads=configs.nheads,
                target_dim=configs.enc_in,
                proj_t=configs.proj_t
            )
            for _ in range(configs.residual_layers)
        ])

    # ── Helpers ─────────────────────────────────────────────────────

    def _np(self, t): return t.detach().cpu().float().numpy()
    def _t(self, a, dev): return torch.tensor(a, dtype=torch.float32, device=dev)

    def _encode(self, coef: torch.Tensor, encoder: TSTransformerEncoder,
                ctx=None) -> torch.Tensor:
        """
        coef : [B, K, L_l]  → encoder → [B, d_model, L_l]
        ctx  : [B, L_ctx, d_model] atau None
        """
        B, K, L_l = coef.shape
        x_in = coef.permute(0, 2, 1)               # [B, L_l, K]
        x_in = torch.cat([x_in, x_in], dim=-1)     # [B, L_l, K*2]  (duplikasi = proxy feat)
        emb  = encoder.encoder(x_in, context_emb=ctx)  # [B, L_l, d_model]
        return emb.permute(0, 2, 1)                 # [B, d_model, L_l]
    
    def _expand_to_KL(self, c: torch.Tensor, K: int, L: int):
        """
        PERBAIKAN 2: ekspansi [B, ch, L] → [B, ch, K*L] dengan MENGULANG
        per variabel, bukan interpolasi lintas batas variabel.
        """
        B, ch, _ = c.shape
        return c.unsqueeze(2).expand(-1, -1, K, -1).reshape(B, ch, K * L)

    def _build_conditions(self, cond_obs: torch.Tensor, K: int, L: int):
        """
        cond_obs : [B, K, L] — observed data
        Return   : [B, channel, K*L]
        """
        ablation = getattr(self.configs, 'ablation', None)

        # ── ABLATION 3: tanpa conditioning sama sekali ─────────────
        if ablation == 'no_conditioning':
            B = cond_obs.shape[0]
            return torch.zeros(B, self.channel, K * L,
                               device=cond_obs.device,
                               dtype=cond_obs.dtype)

        # ── ABLATION 1: tanpa wavelet ──────────────────────────────
        if ablation == 'no_wavelet':
            E = self._encode(cond_obs, self.approx_encoder, ctx=None)   # [B, d, L]
            c = self.branch_projections[0](E)                           # [B, ch, L]
            if self.arch_v2:
                return self._expand_to_KL(c, K, L)
            return F.interpolate(c, size=K * L, mode='linear', align_corners=False)

        # ── Normal / ABLATION 2 ────────────────────────────────────
        arr    = self._np(cond_obs)
        coeffs = _dwt(arr, self.wavelet, self.wav_levels)
        dev    = cond_obs.device

        A   = self._t(coeffs[0], dev)
        E   = self._encode(A, self.approx_encoder, ctx=None)
        ctx = E.permute(0, 2, 1)

        all_embs = [E]
        for det_np, det_enc in zip(coeffs[1:], self.detail_encoders):
            C = self._t(det_np, dev)
            ctx_in = None if ablation == 'no_crosslevel' else ctx
            E_d = self._encode(C, det_enc, ctx=ctx_in)
            all_embs.append(E_d)
            ctx = E_d.permute(0, 2, 1)

        if self.arch_v2:
            # PERBAIKAN 2: upsample hanya sampai L (faktor ~4, bukan ~400)
            cats = []
            for emb, proj in zip(all_embs, self.branch_projections):
                c = proj(emb)                                            # [B, ch, L_l]
                c = F.interpolate(c, size=L, mode='linear', align_corners=False)
                cats.append(c)
            # PERBAIKAN 3: gabung dengan bobot yang dipelajari
            merged = self.branch_merge(torch.cat(cats, dim=1))           # [B, ch, L]
            return self._expand_to_KL(merged, K, L)

        # ── Jalur lama (arch_v1) ───────────────────────────────────
        total = None
        for emb, proj in zip(all_embs, self.branch_projections):
            c = proj(emb)
            c = F.interpolate(c, size=K * L, mode='linear', align_corners=False)
            total = c if total is None else total + c
        return total / len(all_embs)
    
    def forward(self, total_input, side_info, cond_obs, diffusion_step):
        B, inputdim, K, L = total_input.shape

        x = total_input.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channel, K, L)

        # ── HWD: wavelet hierarchical conditioning ───────────────────
        conditions = self._build_conditions(cond_obs, K, L)  # [B, channel, K*L]

        sides = side_info.reshape(B, -1, K * L)
        sides = self.side_projection(sides)             # [B, channel, K*L]

        # Tidak perlu reshape — sudah [B, channel, K*L]
        conditions = conditions + sides
        conditions = self.condition_projection2(conditions)
        conditions = F.relu(conditions)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb, conditions)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channel, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)   # [B, 1, K*L]
        x = x.reshape(B, K, L)
        return x


def _deeper(configs, factor=2):
    """Return config dengan e_layers lebih dalam untuk approx encoder."""
    class _C: pass
    cfg = _C()
    cfg.__dict__.update(vars(configs))
    cfg.e_layers = max(1, configs.e_layers * factor)
    return cfg