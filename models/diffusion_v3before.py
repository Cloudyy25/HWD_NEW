"""
diffusion_v3.py — HWD Adaptive Dual-Domain (wavelet + Fourier + gating)

Base: diffusion.py (skripsi). PERBEDAAN SATU-SATUNYA:
  DWT numpy (pywt.wavedec, tidak differentiable) diganti dengan
  LearnableDWT berbasis PyTorch (konvolusi bertingkat), differentiable.

Dua mode via configs.learnable_wavelet:
  - False (default) : filter beku = db4. Hasil setara skripsi (pywt).
                      → dipakai untuk reproduksi hasil SKRIPSI.
  - True            : filter menjadi nn.Parameter yang dilatih (adaptive
                      wavelet). → fondasi PAPER.

Tambahan dari v2:
  - Cabang Fourier (rfft) sejajar cabang wavelet
  - DualDomainGate: bobot adaptif g per sinyal (wavelet vs fourier)
  - conditioning = g * wavelet + (1-g) * fourier
  configs.use_gating (default True) mengaktifkan mekanisme ini.
File diffusion.py (skripsi) & diffusion_v2.py TIDAK diubah.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

from layers.Diff_layers import TemporalLearning, FeatureLearning
from models.ts_transformer import TSTransformerEncoder


# ── DWT PyTorch differentiable (inline) ─────────────────────────────
def _wavelet_filters(name="db4"):
    w = pywt.Wavelet(name)
    return (torch.tensor(w.dec_lo, dtype=torch.float32),
            torch.tensor(w.dec_hi, dtype=torch.float32))


class LearnableDWT(nn.Module):
    """DWT multi-level differentiable. Mode fixed (db4) atau learnable."""
    def __init__(self, wavelet="db4", levels=3, learnable=False):
        super().__init__()
        self.levels = levels
        self.learnable = learnable
        dec_lo, dec_hi = _wavelet_filters(wavelet)
        self.filt_len = dec_lo.numel()
        if learnable:
            self.dec_lo = nn.Parameter(dec_lo.clone())
            self.dec_hi = nn.Parameter(dec_hi.clone())
        else:
            self.register_buffer("dec_lo", dec_lo)
            self.register_buffer("dec_hi", dec_hi)

    def _dwt1(self, x):
        f = self.filt_len
        lo = self.dec_lo.flip(0).view(1, 1, -1)
        hi = self.dec_hi.flip(0).view(1, 1, -1)
        xp = F.pad(x, (f - 1, 0), mode="circular")
        return F.conv1d(xp, lo, stride=2), F.conv1d(xp, hi, stride=2)

    def forward(self, x):
        """x: [B,K,L] → [approx, det_kasar, ..., det_halus]."""
        B, K, L = x.shape
        max_lvl = pywt.dwt_max_level(L, pywt.Wavelet("db4"))
        levels = min(self.levels, max(1, max_lvl - 1))
        z = x.reshape(B * K, 1, L)
        details, a = [], z
        for _ in range(levels):
            a, d = self._dwt1(a)
            details.append(d)
        approx = a.reshape(B, K, -1)
        dets = [d.reshape(B, K, -1) for d in reversed(details)]
        return [approx] + dets

    def ortho_penalty(self):
        if not self.learnable:
            return torch.zeros((), device=self.dec_lo.device)
        lo = self.dec_lo
        n = lo.numel()
        pen = (torch.dot(lo, lo) - 1.0) ** 2
        for k in range(1, n // 2 + 1):
            sh = torch.zeros_like(lo)
            if 2 * k < n:
                sh[2 * k:] = lo[:n - 2 * k]
            pen = pen + (torch.dot(lo, sh)) ** 2
        return pen


# ── Identik FGTI ────────────────────────────────────────────────────

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# ── Adaptive Dual-Domain Gating (v3) ────────────────────────────────
class _SignalStat(nn.Module):
    """Fitur pembeda transien vs periodik dari [B,K,L]."""
    def forward(self, x):
        Xf = torch.fft.rfft(x, dim=-1).abs()
        Lf = Xf.shape[-1]
        hi = Xf[..., Lf // 2:].pow(2).sum(-1)
        tot = Xf.pow(2).sum(-1) + 1e-8
        hf = hi / tot
        mu = x.mean(-1, keepdim=True); sd = x.std(-1, keepdim=True) + 1e-8
        kurt = (((x - mu) / sd).pow(4).mean(-1))
        e = x.pow(2); conc = e.max(-1).values / (e.mean(-1) + 1e-8)
        return torch.stack([hf, kurt, conc], dim=-1)     # [B,K,3]


class DualDomainGate(nn.Module):
    """Bobot g in [0,1]: tinggi=wavelet, rendah=fourier. Per-batch scalar."""
    def __init__(self, hidden=16):
        super().__init__()
        self.stat = _SignalStat()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        feat = self.stat(x)                  # [B,K,3]
        g = torch.sigmoid(self.net(feat))    # [B,K,1]
        return g.mean(dim=1, keepdim=True)   # [B,1,1] skalar per batch


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
        max_safe = max(1, _pywt.dwt_max_level(configs.seq_len, self.wavelet) - 1)
        self.wav_levels = min(self.wav_levels, max_safe)

        # DWT differentiable (fixed=skripsi, learnable=paper)
        self.learnable_wavelet = getattr(configs, 'learnable_wavelet', False)
        self.dwt = LearnableDWT(self.wavelet, self.wav_levels,
                                learnable=self.learnable_wavelet)

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

        # ── v3: cabang Fourier + gating adaptif ──────────────────────
        self.use_gating = getattr(configs, 'use_gating', True)
        if self.use_gating:
            self.fourier_encoder = TSTransformerEncoder(configs)
            self.fourier_projection = Conv1d_with_init(configs.d_model, self.channel, 1)
            self.gate = DualDomainGate()

        self.side_projection       = Conv1d_with_init(configs.side_dim, self.channel, 1)
        self.condition_projection2 = Conv1d_with_init(self.channel, self.channel, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=configs.diffusion_step_num,
            embedding_dim=configs.d_model
        )

        self.input_projection    = Conv1d_with_init(2, self.channel, 1)
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

    def _build_conditions(self, cond_obs: torch.Tensor, K: int, L: int):
        """
        cond_obs : [B, K, L] — observed data
        Return   : [B, channel, K*L]
        """
        # DWT differentiable (torch) — gradien mengalir ke filter & input
        coeffs = self.dwt(cond_obs)                # list tensor [B, K, L_l]

        # 1. Approx → global context
        A      = coeffs[0]                          # [B, K, L_A]
        E      = self._encode(A, self.approx_encoder, ctx=None)   # [B, d, L_A]
        ctx    = E.permute(0, 2, 1)                # [B, L_A, d]

        all_embs = [E]

        # 2. Detail levels: kasar → halus, tiap level dapat context dari atas
        for i, (C, det_enc) in enumerate(
            zip(coeffs[1:], self.detail_encoders)
        ):
            E_d = self._encode(C, det_enc, ctx=ctx)     # [B, d, L_l]
            all_embs.append(E_d)
            ctx = E_d.permute(0, 2, 1)

        # 3. Project setiap branch: [B, d_model, L_l] → [B, channel, L_l]
        #    lalu upsample ke [B, channel, K*L] — memory efficient
        total = None
        for emb, proj in zip(all_embs, self.branch_projections):
            c = proj(emb)              # [B, channel, L_l]
            c = F.interpolate(
                c, size=K * L,
                mode='linear', align_corners=False
            )                          # [B, channel, K*L]
            total = c if total is None else total + c

        total = total / len(all_embs)          # [B, channel, K*L] — cabang wavelet

        # ── v3: cabang Fourier + fusi gating adaptif ─────────────────
        if self.use_gating:
            # magnitudo spektrum sebagai input encoder Fourier
            Xf = torch.fft.rfft(cond_obs, dim=-1).abs()    # [B, K, Lf]
            E_f = self._encode(Xf, self.fourier_encoder, ctx=None)  # [B, d, Lf]
            c_f = self.fourier_projection(E_f)             # [B, channel, Lf]
            c_f = F.interpolate(c_f, size=K * L, mode='linear', align_corners=False)

            g = self.gate(cond_obs)                        # [B,1,1] bobot wavelet
            total = g * total + (1.0 - g) * c_f            # fusi adaptif

        return total                   # [B, channel, K*L]

    # ── Forward — identik FGTI kecuali conditioning dari wavelet ────

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