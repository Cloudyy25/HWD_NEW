"""
hwd_core.py — logika inti HWD Imputation Explorer (tanpa Streamlit).
Berisi konfigurasi, angka pembanding, rule engine Susenas, mask eksperimen,
metrik, metode pembanding, grafik, serta pemuatan dan inferensi checkpoint HWD.
"""

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
#  LOKASI KODE MODEL
#  Aplikasi mencari folder models/ dan layers/ dari repo HWD di: variabel
#  lingkungan HWD_REPO, folder aplikasi, atau folder induknya.
# ═════════════════════════════════════════════════════════════════════════════
APP_DIR = Path(__file__).resolve().parent
REPO_DIR = None
for _cand in [os.environ.get("HWD_REPO"), APP_DIR, APP_DIR.parent]:
    if _cand and (Path(_cand) / "models").is_dir():
        REPO_DIR = Path(_cand)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        break

SEQ_LEN = 48
SENTINEL = -200.0
RATES = [10, 20, 30, 40]

C_OBS = "#444441"      # nilai asli
C_EVAL = "#C0322F"     # imputasi pada posisi uji
C_NAT = "#BA7517"      # imputasi pada missing alami
C_SPAN = "#F09595"     # pita posisi disembunyikan
C_HWD = "#0F6E56"      # HWD
C_FGTI = "#534AB7"
C_CSDI = "#7F77DD"
C_MEAN = "#888780"
C_MEDIAN = "#B0B0B0"
C_HOTDECK = "#D9A03C"

# ═════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI HWD (Tabel 3.5 dan 3.6)
# ═════════════════════════════════════════════════════════════════════════════
HWD_SPEC = [
    ("Mother wavelet", "Daubechies-2 (db2), filter 4"),
    ("Tingkat dekomposisi", "3 → A₃, D₃, D₂, D₁"),
    ("Panjang koefisien (L = 48)", "8, 8, 14, 25"),
    ("d_model / channel", "128 / 128"),
    ("Encoder A₃ / detail", "8 lapis / 4 lapis × 3"),
    ("Attention head", "8"),
    ("Residual layer", "4"),
    ("Langkah difusi T", "50"),
    ("Jadwal noise", "kuadratik, β 0,0001 → 0,2"),
    ("Optimizer", "Adam, lr 1 × 10⁻³, wd 1 × 10⁻⁶"),
    ("Penjadwal lr", "MultiStepLR 75% dan 90%, γ 0,1"),
    ("Taksiran titik", "median sampel"),
]

DATASET_INFO = pd.DataFrame([
    ["KDD Cup 2018", "Kualitas udara Beijing", 99, "167 (116/51)", 16, 400, 100, 3407],
    ["Guangzhou Traffic", "Kecepatan lalu lintas", 214, "183 (128/55)", 4, 1000, 100, 3407],
    ["PhysioNet 2012", "Rekam medis ICU", 36, "11.988 (8.391/3.597)", 16, 50, 10, 3407],
    ["Susenas", "Konsumsi dan pengeluaran RT", 11, "21.432 (seluruh)", 32, 100, 10, 1],
], columns=["Dataset", "Domain", "K", "Window (latih/uji)", "Batch", "Epoch",
            "Sampel inferensi", "Seed"])

KDD_STATIONS = ["daxing", "fangshan", "huairou", "mentougou", "miyun",
                "pingchang", "pinggu", "shunyi", "tongzhou"]
KDD_VARS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2", "temperature",
            "pressure", "humidity", "wind_direction", "wind_speed"]
KDD_COLS = [f"{s}_{v}" for s in KDD_STATIONS for v in KDD_VARS]
SUSENAS_COLS = ["R101", "R102", "R105", "R301", "R705", "R706", "R707",
                "food", "nonfood", "expend", "kapita"]

# ═════════════════════════════════════════════════════════════════════════════
#  ANGKA PEMBANDING — Yang et al. (2024), mekanisme MCAR
#  MAE dan RMSE dari Tabel 1; CRPS dari Tabel 5 pada lampiran paper FGTI.
# ═════════════════════════════════════════════════════════════════════════════
BASELINES = {
    "KDD Cup 2018": {
        "MAE":  {"Mean": [0.800, 0.796, 0.797, 0.798],
                 "Median": [0.7813,0.7761,0.7767,0.7768],
                 "Hot-deck": [0.1976,0.2039,0.2167,0.2324],
                 "CSDI": [0.177, 0.187, 0.199, 0.220],
                 "FGTI": [0.149, 0.161, 0.176, 0.205]},
        "RMSE": {"Mean": [1.075, 1.071, 1.070, 1.075],
                 "Median":[1.1175,1.1120,1.111,1.1161],
                 "Hot-deck": [0.576,0.581,0.594,0.616],
                 "CSDI": [0.459, 0.500, 0.519, 0.569],
                 "FGTI": [0.406, 0.451, 0.448, 0.478]},
        "CRPS": {"CSDI": [0.224, 0.245, 0.259, 0.278],
                 "FGTI": [0.158, 0.170, 0.186, 0.216]},
    },
    "Guangzhou Traffic": {
        "MAE":  {"Mean": [0.594, 0.594, 0.594, 0.594],
                 "Median": [0.5781,0.5784,0.5781,0.5785],
                 "Hot-deck":[0.2096,0.2163,0.2238,0.2347],
                 "CSDI": [0.210, 0.220, 0.242, 0.283],
                 "FGTI": [0.170, 0.176, 0.202, 0.254]},
        "RMSE": {"Mean": [0.797, 0.799, 0.798, 0.800],
                 "Median":[0.8054,0.8070,0.8063,0.8076],
                 "Hot-deck":[0.3116,0.3253,0.3397,0.3598],
                 "CSDI": [0.306, 0.324, 0.364, 0.439],
                 "FGTI": [0.230, 0.258, 0.291, 0.356]},
        "CRPS": {"CSDI": [0.265, 0.277, 0.292, 0.324],
                 "FGTI": [0.155, 0.168, 0.193, 0.243]},
    },
    "PhysioNet 2012": {
        "MAE":  {"Mean": [0.695, 0.695, 0.696, 0.696],
                 "Median":[0.6774,0.6777,0.6780,0.6780],
                 "Hot-deck":[0.4435,0.4542,0.4635,0.4769],
                 "CSDI": [0.310, 0.335, 0.360, 0.395],
                 "FGTI": [0.286, 0.309, 0.336, 0.376]},
        "RMSE": {"Mean": [0.937, 0.978, 0.987, 0.983],
                 "Median":[1.0016,0.9927,1.0010,0.9969],
                 "Hot-deck":[0.8350,0.8518,0.8541,0.8668],
                 "CSDI": [0.619, 0.664, 0.805, 0.705],
                 "FGTI": [0.580, 0.577, 0.624, 0.669]},
        "CRPS": {"CSDI": [0.544, 0.589, 0.627, 0.671],
                 "FGTI": [0.343, 0.369, 0.389, 0.441]},
    },
}

# Hasil HWD cadangan bila results/hwd_results.csv tidak ada
DEFAULT_HWD_RESULTS = pd.DataFrame([
    ["KDD Cup 2018", "MCAR", 10, "benchmark", "full", 0.1212, 0.4268, 0.1350, "db2, 3 tingkat"],
    ["KDD Cup 2018", "MCAR", 10, "benchmark", "no_wavelet", 0.1253, 0.4304, 0.1387, "ablation"],
    ["KDD Cup 2018", "MCAR", 10, "benchmark", "no_crosslevel", 0.1240, 0.4302, 0.1382, "ablation"],
], columns=["dataset", "mekanisme", "missing_rate", "mode", "konfigurasi",
            "MAE", "RMSE", "CRPS", "catatan"])

ABLATION_LABEL = {"full": "HWD lengkap",
                  "no_wavelet": "Tanpa wavelet conditioning",
                  "no_crosslevel": "Tanpa cross-level attention",
                  "no_conditioning": "Tanpa Conditioning",
                  "guide": "Guide diinterpolasi sepanjang sumbu waktu"}

# ═════════════════════════════════════════════════════════════════════════════
#  SURVEY RULE ENGINE — memakai survey_rules.py dari repo bila tersedia,
#  jika tidak, memakai implementasi cadangan dengan aturan yang sama.
# ═════════════════════════════════════════════════════════════════════════════
FALLBACK_RULES = [
    {"if_col": "_r703_bekerja", "if_val": 1, "null_cols": ["R705"]},
    {"if_col": "_working", "if_val": 0, "null_cols": ["R706", "R707"]},
]


def _is_filled(series):
    """True untuk sel terisi. Aman untuk pandas 2 (object) maupun pandas 3 (str)."""
    filled = series.notna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        filled = filled & (series.astype(str).str.strip() != "")
    return filled


def _fb_prepare(df):
    out = df.copy()
    out["_r703_bekerja"] = (_is_filled(out["R703_A"]).astype(int)
                            if "R703_A" in out.columns else 0)
    r705_yes = pd.Series(False, index=out.index)
    if "R705" in out.columns:
        r705_yes = pd.to_numeric(out["R705"], errors="coerce") == 1
    out["_working"] = ((out["_r703_bekerja"] == 1) | r705_yes).astype(int)
    return out


def _fb_valid_mask(df, rules):
    valid = np.ones(df.shape, dtype=bool)
    col_idx = {c: i for i, c in enumerate(df.columns)}
    for rule in rules:
        if rule["if_col"] not in col_idx:
            continue
        vals = rule["if_val"] if isinstance(rule["if_val"], (list, tuple)) else [rule["if_val"]]
        rows = df[rule["if_col"]].isin(vals).to_numpy()
        for c in rule["null_cols"]:
            if c in col_idx:
                valid[rows, col_idx[c]] = False
    return valid


def _fb_restore(df_imp, df_asli_prep, rules):
    out = df_imp.copy()
    valid = _fb_valid_mask(df_asli_prep, rules)
    for j, c in enumerate(df_asli_prep.columns):
        if c in out.columns:
            out.loc[~valid[:, j], c] = np.nan
    return out


try:
    import survey_rules as _sr  # noqa: E402
    RULES = _sr.SUSENAS_SKIP_RULES
    prepare_susenas = _sr.prepare_susenas_columns
    valid_mask_fn = lambda d: _sr.get_valid_mask(d, RULES)          # noqa: E731
    restore_fn = lambda imp, asli: _sr.restore_structural(imp, asli, RULES)  # noqa: E731
    RULE_SOURCE = "survey_rules.py (repo)"
except Exception:
    RULES = FALLBACK_RULES
    prepare_susenas = _fb_prepare
    valid_mask_fn = lambda d: _fb_valid_mask(d, RULES)              # noqa: E731
    restore_fn = lambda imp, asli: _fb_restore(imp, asli, RULES)    # noqa: E731
    RULE_SOURCE = "implementasi cadangan di app.py (aturan sama)"

# ═════════════════════════════════════════════════════════════════════════════
#  UTILITAS
# ═════════════════════════════════════════════════════════════════════════════


def idn(x, nd=4):
    """Format angka dengan koma desimal; '–' bila kosong."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def load_npy(uploaded):
    return np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False)


def default_col_names(K):
    if K == len(KDD_COLS):
        return KDD_COLS
    if K == len(SUSENAS_COLS):
        return SUSENAS_COLS
    return [f"var_{i}" for i in range(K)]


def make_windows(Xn):
    n_ok = (len(Xn) // SEQ_LEN) * SEQ_LEN
    data = Xn[:n_ok].reshape(-1, SEQ_LEN, Xn.shape[1])
    return data, n_ok


def make_mask(data, gt_mask, mechanism, rate, seed, exclude_idx=(), mar_col=None, block_len=4):
    """Mask eksperimen; salinan logika notebook Susenas (make_mask)."""
    rng = np.random.default_rng(seed)
    N, L, K = data.shape
    cond = gt_mask.copy()
    obs_idx = np.argwhere(gt_mask == 1)
    if len(exclude_idx):
        obs_idx = obs_idx[~np.isin(obs_idx[:, 2], list(exclude_idx))]
    n_hide = int(len(obs_idx) * rate)
    if n_hide == 0:
        return cond
    if mechanism == "MCAR":
        pick = obs_idx[rng.choice(len(obs_idx), n_hide, replace=False)]
        cond[pick[:, 0], pick[:, 1], pick[:, 2]] = 0.0
    elif mechanism == "MAR":
        fval, fobs = data[:, :, mar_col], gt_mask[:, :, mar_col]
        fcnt = fobs.sum(axis=1)
        fmean = np.where(fcnt > 0, (fval * fobs).sum(axis=1) / np.maximum(fcnt, 1), np.nan)
        fmean[np.isnan(fmean)] = np.nanmin(fmean) if np.isfinite(np.nanmin(fmean)) else 0.0
        rank = (np.argsort(np.argsort(fmean)) + 1).astype(float)
        weight = np.repeat(rank / rank.sum(), L * K).reshape(N, L, K) * gt_mask
        if len(exclude_idx):
            weight[:, :, list(exclude_idx)] = 0.0
        weight = weight.flatten()
        weight = weight / weight.sum()
        u = rng.random(weight.shape)
        keys = np.full(weight.shape, np.inf)          # bobot nol tidak pernah terpilih
        pos = weight > 0
        keys[pos] = -np.log(u[pos] + 1e-12) / weight[pos]
        chosen = np.argpartition(keys, n_hide)[:n_hide]
        flat = cond.flatten()
        flat[chosen] = 0.0
        cond = flat.reshape(N, L, K)
    else:  # block
        maskable = [c for c in range(K) if c not in set(exclude_idx)]
        hidden, guard = 0, 0
        while hidden < n_hide and guard < n_hide * 50:
            guard += 1
            w = rng.integers(0, N)
            c = maskable[rng.integers(0, len(maskable))]
            s = rng.integers(0, max(1, L - block_len + 1))
            for t in range(s, min(s + block_len, L)):
                if gt_mask[w, t, c] == 1 and cond[w, t, c] == 1:
                    cond[w, t, c] = 0.0
                    hidden += 1
    return cond


def mae_rmse(true, pred, mask):
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.nan, np.nan
    d = pred[m] - true[m]
    return float(np.abs(d).mean()), float(np.sqrt((d ** 2).mean()))


def plot_series(data, pred, eval_mask, cond_mask, w, var_idx, names, ylabel="Z-score"):
    """Grafik satu window: nilai asli, pita posisi uji, imputasi uji, imputasi missing alami."""
    n = len(var_idx)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True, squeeze=False)
    t = np.arange(data.shape[1])
    for ax, k in zip(axes[:, 0], var_idx):
        obs = np.where(data[w, :, k] <= SENTINEL + 10, np.nan, data[w, :, k])
        ev = eval_mask[w, :, k].astype(bool)
        nat = (cond_mask[w, :, k] == 0) & ~ev & np.isnan(obs) if cond_mask is not None else np.zeros_like(ev)
        for s in np.where(ev)[0]:
            ax.axvspan(s - 0.5, s + 0.5, color=C_SPAN, alpha=0.25, lw=0)
        ax.plot(t, obs, color=C_OBS, lw=1.4, label="Nilai asli", zorder=3)
        ax.plot(t, np.where(ev, pred[w, :, k], np.nan), "o", color=C_EVAL, ms=5,
                label="Imputasi (posisi uji)", zorder=5)
        if nat.any():
            ax.plot(t, np.where(nat, pred[w, :, k], np.nan), "^", color=C_NAT, ms=5,
                    alpha=0.85, label="Imputasi (missing alami)", zorder=4)
        ax.set_title(names[k], fontsize=9, loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.25, ls="--", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1, 0].set_xlabel("Posisi dalam window (0–47)", fontsize=9)
    h, lab = [], []
    for ax in axes[:, 0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    h.append(plt.Rectangle((0, 0), 1, 1, color=C_SPAN, alpha=0.25))
    lab.append("Posisi disembunyikan")
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_scatter(true, pred, title, max_points=3000, seed=0):
    rng = np.random.default_rng(seed)
    if len(true) > max_points:
        idx = rng.choice(len(true), max_points, replace=False)
        true, pred = true[idx], pred[idx]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(true, pred, s=6, alpha=0.3, color=C_EVAL, edgecolors="none")
    lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    pad = (hi - lo) * 0.05 or 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=C_OBS, lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    r = np.corrcoef(true, pred)[0, 1] if len(true) > 2 else np.nan
    ax.set_title(f"{title}\nr = {idn(r, 3)}  ·  n = {len(true):,}".replace(",", "."), fontsize=9)
    ax.set_xlabel("Nilai asli", fontsize=8)
    ax.set_ylabel("Imputasi", fontsize=8)
    ax.grid(alpha=0.2, ls="--")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


# ── HWD: konfigurasi & inferensi (mengikuti notebook Susenas) ─────────────────

def hwd_config(K, ck=None):
    ck = ck or {}
    return SimpleNamespace(
        seq_len=SEQ_LEN, enc_in=K, c_out=K,
        d_model=128, e_layers=4, nheads=8, channel=128, proj_t=128,
        residual_layers=4, timeemb=128, featureemb=16,
        diffusion_step_num=50, schedule="quad", beta_start=1e-4, beta_end=0.2,
        epoch_diff=100, learning_rate_diff=1e-3,
        mask_ratio_ssl=0.2, avg_mask_len_ssl=3,
        wavelet=ck.get("wavelet", "db2"), levels=3, batch=8,
        device="cpu", missing_rate=0.1, seed=1, n_samples=10,
        mechanism=str(ck.get("mechanism", "mcar")),
        ablation=ck.get("ablation", None), arch_v2=bool(ck.get("arch_v2", False)),
    )


def load_hwd_checkpoint(ckpt_bytes, K):
    import torch
    from models import main_model
    ck = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    meta = ck if isinstance(ck, dict) else {}
    if "embed_layer.weight" in sd and sd["embed_layer.weight"].shape[0] != K:
        raise ValueError(f"Checkpoint dilatih untuk K = {sd['embed_layer.weight'].shape[0]} "
                         f"variabel, sedangkan data memiliki K = {K}.")
    if meta.get("wavelet", "db2") != "db2":
        raise ValueError(f"Checkpoint memakai wavelet {meta.get('wavelet')}; aplikasi ini untuk db2.")
    cfg = hwd_config(K, meta)
    model = main_model.HWD(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise ValueError(f"{len(missing)} bobot tidak ditemukan di checkpoint "
                         f"(contoh: {missing[0]}). Arsitektur tidak cocok.")
    model.eval()
    return model, meta, unexpected


def run_hwd(model, data, cond, gt, n_samples, bs, progress=None):
    """Mengembalikan median [N, L, K] dan sampel posisi uji untuk CRPS."""
    import torch
    N = data.shape[0]
    med_all, crps_parts = [], []
    try:
        import properscoring as ps
    except Exception:
        ps = None
    n_batch = int(np.ceil(N / bs))
    for b in range(n_batch):
        sl = slice(b * bs, min((b + 1) * bs, N))
        d = torch.from_numpy(data[sl]).float()
        c = torch.from_numpy(cond[sl]).float()
        g = torch.from_numpy(gt[sl]).float()
        tp = torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(0).repeat(d.shape[0], 1)
        with torch.no_grad():
            samples, obs, ev, _, _ = model.evaluate(d, c, tp, g, n_samples=n_samples)
        med = samples.median(dim=1).values                     # [B, K, L]
        med_all.append(med.permute(0, 2, 1).cpu().numpy())
        if ps is not None:
            m = ev.cpu().numpy().astype(bool)
            if m.any():
                s_np = np.moveaxis(samples.cpu().numpy(), 1, -1)   # [B, K, L, n]
                crps_parts.append(ps.crps_ensemble(obs.cpu().numpy()[m], s_np[m]))
        if progress is not None:
            progress((b + 1) / n_batch, f"Batch {b + 1}/{n_batch}")
    crps = float(np.concatenate(crps_parts).mean()) if crps_parts else np.nan
    return np.concatenate(med_all, axis=0), crps


def baseline_mean(data, cond):
    """Mean imputation: rata-rata tiap variabel dari posisi kondisi."""
    vals = np.where(cond == 1, data, np.nan).reshape(-1, data.shape[2])
    mu = np.nanmean(vals, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    return np.broadcast_to(mu, data.shape).copy()


def baseline_interp(data, cond):
    """Interpolasi linear per variabel sepanjang urutan baris, dari posisi kondisi."""
    flat = np.where(cond == 1, data, np.nan).reshape(-1, data.shape[2])
    df = pd.DataFrame(flat).interpolate(limit_direction="both")
    out = df.fillna(0.0).to_numpy()
    return out.reshape(data.shape)




# ═════════════════════════════════════════════════════════════════════════════
#  IMPUTASI SUSENAS — paket model, praproses, keluaran, metrik, grafik
# ═════════════════════════════════════════════════════════════════════════════
SUSENAS_EXPEND = ["food", "nonfood", "expend", "kapita"]
SUSENAS_ID = ["R101", "R102"]
SUSENAS_CODE_VARS = ["R101", "R102", "R105", "R301", "R705", "R706", "R707"]
SUSENAS_LABEL = {"food": "Pengeluaran Makanan", "nonfood": "Pengeluaran Bukan Makanan",
                 "expend": "Total Pengeluaran Rumah Tangga", "kapita": "Pengeluaran per Kapita"}


def match_columns(df, names):
    """Cocokkan nama kolom tanpa membedakan huruf besar-kecil."""
    lower = {str(c).lower(): c for c in df.columns}
    found, missing = [], []
    for n in names:
        c = lower.get(str(n).lower())
        if c is None:
            missing.append(n)
        else:
            found.append(c)
    return found, missing


def susenas_prepare(df, paket):
    """Praproses data unggahan dengan parameter tahun latih (means, stds, batas 3×IQR),
    lalu padding ke kelipatan panjang window."""
    feats, missing = match_columns(df, paket["feat_cols"])
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di data: {missing}")
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n, K = X.shape
    means = np.asarray(paket["means"], dtype=float)
    stds = np.asarray(paket["stds"], dtype=float)
    stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds)
    if len(means) != K:
        raise ValueError(f"Paket model berisi {len(means)} variabel, data memiliki {K}.")
    nat = np.isnan(X)
    out = np.zeros_like(nat)
    lower_feats = [str(f).lower() for f in paket["feat_cols"]]
    for name, (lo, hi) in paket.get("fences", {}).items():
        j = lower_feats.index(str(name).lower())
        v = X[:, j]
        out[:, j] = ~np.isnan(v) & ((v < lo) | (v > hi))
    Xn = (np.where(out, np.nan, X) - means) / stds
    Xn[np.isnan(Xn)] = SENTINEL
    L = int(paket.get("seq_len", SEQ_LEN))
    W = int(np.ceil(n / L))
    pad = W * L - n
    if pad:
        Xn = np.vstack([Xn, np.full((pad, K), SENTINEL)])
    data = Xn.reshape(W, L, K)
    gt = (data != SENTINEL).astype(np.float32)
    return {"feats": feats, "X": X, "nat": nat, "out": out, "data": data, "gt": gt,
            "n": n, "pad": pad, "means": means, "stds": stds, "L": L}


def susenas_mask(prep, mechanism, rate, seed, block_len=4):
    """Mask eksperimen pada posisi teramati, kolom identitas dikecualikan."""
    names = [str(f).lower() for f in prep["feats"]]
    ex = [names.index(c.lower()) for c in SUSENAS_ID if c.lower() in names]
    mar_col = names.index("food") if "food" in names else None
    return make_mask(prep["data"], prep["gt"], mechanism, rate, int(seed), ex, mar_col, block_len)


def _snap(values, valid):
    valid = np.asarray(sorted(set(valid)))
    if valid.size == 0:
        return values
    idx = np.abs(values[:, None] - valid[None, :]).argmin(axis=1)
    return valid[idx]


def susenas_compose(df, prep, pred_z, cond, replace_outlier=False, snap_codes=True):
    """Susun berkas keluaran: nilai kondisi tetap, posisi kosong diisi hasil imputasi,
    outlier dipertahankan kecuali diminta diganti, lalu structural missing dikosongkan."""
    K, n = len(prep["feats"]), prep["n"]
    pred = pred_z.reshape(-1, K)[:n] * prep["stds"] + prep["means"]
    condf = cond.reshape(-1, K)[:n].astype(bool)
    fill = ~condf
    if not replace_outlier:
        fill &= ~prep["out"]
    X = prep["X"]
    outX = np.where(fill, pred, X)
    if snap_codes:
        code_lower = {c.lower() for c in SUSENAS_CODE_VARS}
        for j, f in enumerate(prep["feats"]):
            if str(f).lower() in code_lower and fill[:, j].any():
                valid = X[:, j][~np.isnan(X[:, j])]
                outX[fill[:, j], j] = _snap(outX[fill[:, j], j], valid)
    exp_lower = {c.lower() for c in SUSENAS_EXPEND}
    for j, f in enumerate(prep["feats"]):
        if str(f).lower() in exp_lower:                  # pengeluaran tidak boleh negatif
            outX[fill[:, j], j] = np.clip(outX[fill[:, j], j], 0, None)
    out_df = df.reset_index(drop=True).copy()
    for j, f in enumerate(prep["feats"]):
        out_df[f] = outX[:, j]
    n_struct = 0
    rule_cols = {"R703_A", "R705", "R706", "R707"}
    if rule_cols.issubset({str(c).upper() for c in df.columns}):
        asli = prepare_susenas(df.reset_index(drop=True))
        restored = restore_fn(out_df, asli)
        n_struct = int((out_df[prep["feats"]].notna().to_numpy() & restored[prep["feats"]].isna().to_numpy()).sum())
        out_df = restored[list(df.columns)]
    final = out_df[prep["feats"]].to_numpy(dtype=float)
    imputed = fill & ~np.isnan(final)
    feats = list(prep["feats"])
    out_df["_diimputasi"] = [", ".join(feats[j] for j in np.flatnonzero(r)) for r in imputed]
    out_df["_outlier"] = [", ".join(feats[j] for j in np.flatnonzero(r)) for r in prep["out"]]
    return out_df, {"diimputasi": int(imputed.sum()), "outlier": int(prep["out"].sum()),
                    "structural": n_struct, "kosong_awal": int((prep["nat"]).sum())}


def susenas_metrics(prep, pred_z, ev):
    """Metrik pada posisi uji: skala Z (semua variabel) dan rupiah (variabel pengeluaran)."""
    K, n = len(prep["feats"]), prep["n"]
    t = prep["data"].reshape(-1, K)[:n]
    p = pred_z.reshape(-1, K)[:n]
    e = ev.reshape(-1, K)[:n].astype(bool)
    d = p[e] - t[e]
    res = {"MAE_z": float(np.abs(d).mean()) if d.size else np.nan,
           "RMSE_z": float(np.sqrt((d ** 2).mean())) if d.size else np.nan, "n_uji": int(e.sum())}
    rows, pairs, eval_rows = [], {}, []
    names = [str(f).lower() for f in prep["feats"]]
    for j, f in enumerate(prep["feats"]):
        m = e[:, j]
        if not m.any():
            continue
        y = t[m, j] * prep["stds"][j] + prep["means"][j]
        yh = p[m, j] * prep["stds"][j] + prep["means"][j]
        for r, a, b in zip(np.flatnonzero(m), y, yh):
            eval_rows.append({"baris": int(r) + 1, "variabel": f, "nilai_asli": a,
                              "imputasi_hwd": b, "galat_absolut": abs(b - a)})
        if names[j] in SUSENAS_EXPEND:
            nz = np.abs(y) > 0
            rows.append({"Variabel": f, "n": int(m.sum()), "MAE (Rp)": float(np.abs(yh - y).mean()),
                         "RMSE (Rp)": float(np.sqrt(((yh - y) ** 2).mean())),
                         "MAPE (%)": float(np.mean(np.abs((yh[nz] - y[nz]) / y[nz])) * 100) if nz.any() else np.nan})
            pairs[names[j]] = (y, yh)
    return res, pd.DataFrame(rows), pairs, pd.DataFrame(eval_rows)


def plot_susenas_scatter(pairs, max_points=1000, seed=0):
    """Scatter 2×2 seperti Gambar 4.8–4.14: sumbu X = sampel ke-, sumbu Y = rupiah."""
    from matplotlib.ticker import FuncFormatter
    rp = FuncFormatter(lambda v, _: f"Rp{v/1e6:,.1f} jt".replace(",", "X").replace(".", ",").replace("X", ".")
                       if abs(v) >= 1e6 else f"Rp{v/1e3:,.0f} rb".replace(",", "."))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, var in zip(axes.flat, SUSENAS_EXPEND):
        if var not in pairs:
            ax.set_axis_off()
            ax.set_title(f"{SUSENAS_LABEL[var]}: tidak ada posisi uji", fontsize=9)
            continue
        y, yh = pairs[var]
        if len(y) > max_points:
            idx = rng.choice(len(y), max_points, replace=False)
            y, yh = y[idx], yh[idx]
        x = np.arange(1, len(y) + 1)
        ax.scatter(x, y, s=16, color="#185FA5", alpha=0.75, label="Nilai asli", edgecolors="none")
        ax.scatter(x, yh, s=16, color="#C0322F", alpha=0.75, label="Imputasi HWD", edgecolors="none")
        r = np.corrcoef(y, yh)[0, 1] if len(y) > 2 else np.nan
        nz = np.abs(y) > 0
        mape = np.mean(np.abs((yh[nz] - y[nz]) / y[nz])) * 100 if nz.any() else np.nan
        mae = np.abs(yh - y).mean()
        ax.text(0.02, 0.97, f"r = {idn(r, 3)}\nMAE = Rp{mae:,.0f}".replace(",", ".") +
                f"\nMAPE = {idn(mape, 2)}%\nn = {len(y)}", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))
        ax.set_title(SUSENAS_LABEL[var], fontsize=10)
        ax.set_xlabel("Sampel ke-", fontsize=8)
        ax.yaxis.set_major_formatter(rp)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2, ls="--")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, lab = axes.flat[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig
