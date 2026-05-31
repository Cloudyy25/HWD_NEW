"""
train.py — HWD
==============
Menggabungkan A_diffusion_train.py + A_train.py dari FGTI menjadi satu file.

Perubahan dari FGTI:
  - Tambah preprocess_kdd() dan preprocess_imk() lengkap di sini
  - Hapus observed_dataf dari semua loop
  - Import HWD bukan FGTI dari main_model
  - Ganti --flimit / --topf dengan --wavelet / --levels
  - Tambah CRPS via utils.calc_quantile_CRPS
  - get_config() bisa dipanggil dari Colab tanpa argparse
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from torch import optim

from models import main_model
import dataset as A_dataset
import utils


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# Fungsi ini dipanggil SEKALI sebelum training untuk menghasilkan KDD_norm.csv.
# Setelah file CSV ada, fungsi ini skip otomatis (ada guard os.path.exists).
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_kdd(raw='Data/KDD.csv', out='Data/KDD_norm.csv'):
    """
    Preprocessing KDD Cup 2018:
      1. Sentinel ≥ 999000  → NaN   (kode sensor rusak di wind_direction)
      2. Outlier  3×IQR     → NaN   (nilai ekstrem per kolom)
      3. NaN                → -200  (missing marker untuk model)
      4. Z-score normalisasi dari clean values
      5. Simpan KDD_norm.csv, KDD_means.npy, KDD_stds.npy
    """
    if os.path.exists(out):
        print(f'Skip — {out} sudah ada')
        return

    df   = pd.read_csv(raw, header=0)
    data = df.select_dtypes(include=[np.number]).to_numpy().astype(float)
    N, K = data.shape

    # ── Step 1: Sentinel → NaN ───────────────────────────────────────────────
    n_sent = (data >= 999000).sum()
    data[data >= 999000] = np.nan
    print(f'[preprocess_kdd] Sentinel ≥999000 → NaN : {n_sent} sel')

    # ── Step 2: Outlier 3×IQR → NaN ─────────────────────────────────────────
    n_out = 0
    for j in range(K):
        col   = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 10:
            continue
        q1, q3 = np.percentile(valid, 25), np.percentile(valid, 75)
        iqr    = q3 - q1
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        mask_out = (~np.isnan(col)) & ((col < lo) | (col > hi))
        n_out   += mask_out.sum()
        data[mask_out, j] = np.nan
    print(f'[preprocess_kdd] Outlier 3×IQR     → NaN : {n_out} sel')

    # ── Step 3: NaN → -200 ───────────────────────────────────────────────────
    total_missing = np.isnan(data).sum()
    print(f'[preprocess_kdd] Total → -200       : {total_missing} sel '
          f'({total_missing / (N * K) * 100:.2f}%)')
    data[np.isnan(data)] = -200

    # ── Step 4: Z-score normalisasi ──────────────────────────────────────────
    means, stds = [], []
    for j in range(K):
        obs = data[data[:, j] != -200, j]
        if len(obs) == 0:
            means.append(0); stds.append(1); continue
        m, s = obs.mean(), obs.std() + 1e-8
        data[data[:, j] != -200, j] = (data[data[:, j] != -200, j] - m) / s
        means.append(m); stds.append(s)

    # ── Step 5: Simpan ───────────────────────────────────────────────────────
    np.savetxt(out, data, delimiter=',', fmt='%6f')
    np.save('Data/KDD_means.npy', np.array(means))
    np.save('Data/KDD_stds.npy',  np.array(stds))
    print(f'[preprocess_kdd] Saved: {out}  shape={data.shape}')


def preprocess_imk(raw='Data/IMK_raw.csv', out='Data/IMK_norm.csv'):
    """
    Preprocessing IMK BPS:
      1. Outlier 3×IQR → NaN  (tidak ada sentinel khusus seperti KDD)
         Lower bound di-clamp ke 0 untuk variabel non-negatif (nilai ekonomi)
      2. NaN            → -200
      3. Z-score normalisasi
      4. Simpan IMK_norm.csv, IMK_means.npy, IMK_stds.npy
    """
    if os.path.exists(out):
        print(f'Skip — {out} sudah ada')
        return

    df   = pd.read_csv(raw, header=0)
    data = df.select_dtypes(include=[np.number]).to_numpy().astype(float)
    N, K = data.shape

    # ── Step 1: Outlier 3×IQR → NaN ─────────────────────────────────────────
    n_out = 0
    for j in range(K):
        col   = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 10:
            continue
        q1, q3 = np.percentile(valid, 25), np.percentile(valid, 75)
        iqr    = q3 - q1
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        if q1 >= 0:           # variabel non-negatif: lower fence min 0
            lo = max(lo, 0)
        mask_out = (~np.isnan(col)) & ((col < lo) | (col > hi))
        n_out   += mask_out.sum()
        data[mask_out, j] = np.nan
    print(f'[preprocess_imk] Outlier 3×IQR     → NaN : {n_out} sel')

    # ── Step 2: NaN → -200 ───────────────────────────────────────────────────
    total_missing = np.isnan(data).sum()
    print(f'[preprocess_imk] Total → -200       : {total_missing} sel '
          f'({total_missing / (N * K) * 100:.2f}%)')
    data[np.isnan(data)] = -200

    # ── Step 3: Z-score normalisasi ──────────────────────────────────────────
    means, stds = [], []
    for j in range(K):
        obs = data[data[:, j] != -200, j]
        if len(obs) == 0:
            means.append(0); stds.append(1); continue
        m, s = obs.mean(), obs.std() + 1e-8
        data[data[:, j] != -200, j] = (data[data[:, j] != -200, j] - m) / s
        means.append(m); stds.append(s)

    # ── Step 4: Simpan ───────────────────────────────────────────────────────
    np.savetxt(out, data, delimiter=',', fmt='%6f')
    np.save('Data/IMK_means.npy', np.array(means))
    np.save('Data/IMK_stds.npy',  np.array(stds))
    print(f'[preprocess_imk] Saved: {out}  shape={data.shape}')


def preprocess_generic(raw_csv: str, out_csv: str, prefix: str = 'Data'):
    """
    Preprocessing generik untuk dataset selain KDD dan IMK:
    hanya NaN → -200 dan z-score. Tidak ada sentinel atau outlier treatment
    karena Guangzhou dan PhysioNet tidak punya sentinel value seperti KDD.
    """
    if os.path.exists(out_csv):
        print(f'Skip — {out_csv} sudah ada')
        return

    df   = pd.read_csv(raw_csv, header=0)
    data = df.select_dtypes(include=[np.number]).to_numpy().astype(float)
    N, K = data.shape

    data[np.isnan(data)] = -200

    means, stds = [], []
    for j in range(K):
        obs = data[data[:, j] != -200, j]
        if len(obs) == 0:
            means.append(0); stds.append(1); continue
        m, s = obs.mean(), obs.std() + 1e-8
        data[data[:, j] != -200, j] = (data[data[:, j] != -200, j] - m) / s
        means.append(m); stds.append(s)

    np.savetxt(out_csv, data, delimiter=',', fmt='%6f')
    data_dir = os.path.dirname(out_csv) or 'Data'
    np.save(f'{data_dir}/{prefix}_means.npy', np.array(means))
    np.save(f'{data_dir}/{prefix}_stds.npy',  np.array(stds))
    print(f'[preprocess_generic] Saved: {out_csv}  shape={data.shape}')



# ══════════════════════════════════════════════════════════════════════════════
# get_config — bisa dipanggil dari Colab (override=dict) atau CLI (override=None)
# ══════════════════════════════════════════════════════════════════════════════

def get_config(override: dict = None):
    """
    Return config namespace.
    Kalau override=None  → parse dari sys.argv (mode CLI).
    Kalau override=dict  → pakai dict itu (mode Colab notebook).

    Contoh dari Colab:
        cfg = train.get_config({
            "dataset": "kdd",
            "missing_rate": 0.1,
            "epoch_diff": 200,
        })
    """
    parser = argparse.ArgumentParser(description='HWD')

    # ── Data & general ───────────────────────────────────────────────────────
    parser.add_argument('--device',       type=str,   default='cuda')
    parser.add_argument('--batch',        type=int,   default=16)
    parser.add_argument('--dataset',      type=str,   default='kdd',
                        help='kdd | guangzhou | physio | imk')
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--seed',         type=int,   default=3407)

    # ── Dataset shape ────────────────────────────────────────────────────────
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--enc_in',  type=int, default=99)   # KDD: 99
    parser.add_argument('--c_out',   type=int, default=99)

    # ── Encoder ──────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',  type=int, default=128)
    parser.add_argument('--e_layers', type=int, default=4)

    # ── Diffusion ────────────────────────────────────────────────────────────
    parser.add_argument('--diffusion_step_num', type=int,   default=50)
    parser.add_argument('--timeemb',            type=int,   default=128)
    parser.add_argument('--featureemb',         type=int,   default=16)
    parser.add_argument('--nheads',             type=int,   default=8)
    parser.add_argument('--channel',            type=int,   default=128)
    parser.add_argument('--proj_t',             type=int,   default=128)
    parser.add_argument('--residual_layers',    type=int,   default=4)
    parser.add_argument('--schedule',           type=str,   default='quad')
    parser.add_argument('--beta_start',         type=float, default=0.0001)
    parser.add_argument('--beta_end',           type=float, default=0.2)
    parser.add_argument('--epoch_diff',         type=int,   default=200)
    parser.add_argument('--learning_rate_diff', type=float, default=1e-3)

    # ── SSL masking ──────────────────────────────────────────────────────────
    parser.add_argument('--mask_ratio_ssl',   type=float, default=0.2)
    parser.add_argument('--avg_mask_len_ssl', type=int,   default=3)

    # ── HWD: wavelet ─────────────────────────────────────────────────────────
    parser.add_argument('--wavelet', type=str, default='db4')
    parser.add_argument('--levels',  type=int, default=3)

    # ── IMK-specific ─────────────────────────────────────────────────────────
    parser.add_argument('--survey_config',   type=str, default='')
    parser.add_argument('--col_names_path',  type=str, default='',
                        help='Path CSV dengan nama kolom asli (untuk IMK survey rules)')
    parser.add_argument('--data_path',     type=str, default='')

    # ── Incremental learning ─────────────────────────────────────────────────
    parser.add_argument('--incremental',  action='store_true')
    parser.add_argument('--base_ckpt',    type=str,   default='')
    parser.add_argument('--freeze_ratio', type=float, default=0.8)
    parser.add_argument('--incr_lr',      type=float, default=1e-5)
    parser.add_argument('--incr_epochs',  type=int,   default=50)

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument('--save_dir',  type=str, default='Checkpoints/')
    parser.add_argument('--n_samples', type=int, default=100)

    if override is not None:
        configs = parser.parse_args([])
        for k, v in override.items():
            setattr(configs, k, v)
    else:
        configs = parser.parse_args()

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def diffusion_train(configs):
    train_loader, test_loader = A_dataset.get_dataset(configs)

    model = main_model.HWD(configs).to(configs.device)

    model_optim = optim.Adam(
        model.parameters(),
        lr=configs.learning_rate_diff,
        weight_decay=1e-6
    )
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.90 * configs.epoch_diff)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )

    for epoch in range(configs.epoch_diff):
        train_loss = []
        epoch_time = time.time()

        model.train()
        for observed_data, observed_mask, observed_tp, gt_mask in train_loader:
            model_optim.zero_grad()
            loss = model(observed_data, observed_mask, observed_tp, gt_mask)
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())

        lr_scheduler.step()

        if epoch % 50 == 0 or epoch == configs.epoch_diff - 1:
            print(f"Epoch {epoch+1:>4} | time {time.time()-epoch_time:.1f}s "
                  f"| loss {np.average(train_loss):.6f}")

    # Simpan checkpoint
    os.makedirs(configs.save_dir, exist_ok=True)
    ckpt_path = os.path.join(
        configs.save_dir,
        f"hwd_{configs.dataset}_mr{configs.missing_rate}.pt"
    )
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    return model



# ══════════════════════════════════════════════════════════════════════════════
# Helper: nama kolom numerik per dataset (untuk header CSV output)
# ══════════════════════════════════════════════════════════════════════════════

def get_num_cols(dataset: str) -> list:
    """Return nama kolom numerik untuk dataset tertentu, dipakai sebagai
    header CSV hasil imputasi agar kolom terbaca jelas."""
    csv_map = {
        'kdd':       'Data/KDD.csv',
        'imk':       'Data/IMK_raw.csv',
        'guangzhou': 'Data/guangzhou.csv',
        'physio':    'Data/physio.csv',
    }
    path = csv_map.get(dataset, '')
    if path and os.path.exists(path):
        df = pd.read_csv(path, header=0, nrows=0)   # baca header saja
        # exclude='object' lebih robust dari include=[np.number] di pandas 2.x/3.x
        return df.select_dtypes(exclude=['object']).columns.tolist()
    return []   # fallback: tanpa header (pakai index numerik)

def full_inference_and_save(configs, model):
    """
    Jalankan HWD inference pada SELURUH data (train + test, semua 8016 baris).
    Setiap posisi -200 (natural missing + artificial missing) diisi dengan
    prediksi HWD — tidak ada mean fallback.

    Output: Data/{DATASET}_HWD_full_mr{rate}.csv
      - Semua baris (8016) × semua kolom (numerik + kategorik)
      - Tidak ada NaN di kolom numerik
      - 1 kolom audit: _is_test (1=test rows, 0=train rows)
    """
    import os

    dataset  = configs.dataset
    rate     = configs.missing_rate
    seq_len  = configs.seq_len

    # ── Map dataset ke raw CSV (untuk ambil kategorik + nama kolom) ──────────
    raw_map = {
        'kdd'      : 'Data/KDD.csv',
        'guangzhou': 'Data/guangzhou.csv',
        'physio'   : 'Data/Physio_norm.csv',
        'imk'      : getattr(configs, 'col_names_path', ''),
    }
    norm_map = {
        'kdd'      : 'Data/KDD_norm.csv',
        'guangzhou': 'Data/Guangzhou_norm.csv',
        'physio'   : 'Data/Physio_norm.csv',
        'imk'      : 'Data/IMK_norm.csv',
    }
    raw_csv  = raw_map.get(dataset, '')
    norm_csv = norm_map.get(dataset, '')
    means_path = f'Data/{dataset.upper()}_means.npy'
    stds_path  = f'Data/{dataset.upper()}_stds.npy'

    if not os.path.exists(norm_csv):
        print(f"[full_inference] {norm_csv} tidak ada — skip"); return
    if not os.path.exists(means_path):
        print(f"[full_inference] {means_path} tidak ada — skip"); return

    # ── Load data norm dan means/stds ────────────────────────────────────────
    data_norm  = np.loadtxt(norm_csv, delimiter=',')   # [N, K]
    means      = np.load(means_path)
    stds       = np.load(stds_path)
    N, K       = data_norm.shape
    n_windows  = N // seq_len
    n_used     = n_windows * seq_len                   # baris yang masuk window

    # Posisi yang perlu diimputasi: semua yang -200
    mask_missing = (data_norm == -200)                 # [N, K] bool

    # ── Jalankan inference di setiap window (train + test) ───────────────────
    model.eval()
    data_w    = data_norm[:n_used].reshape(n_windows, seq_len, K)   # [W, L, K]
    miss_w    = mask_missing[:n_used].reshape(n_windows, seq_len, K)
    obs_w     = (~miss_w).astype(np.float32)                         # 1=observed
    result_w  = data_w.copy()                                        # akan diisi

    batch_size = configs.batch * 2   # inference lebih cepat dari training
    n_batches  = (n_windows + batch_size - 1) // batch_size

    print(f"[full_inference] Running on all {n_windows} windows "
          f"({n_batches} batches)...")

    import torch
    with torch.no_grad():
        for b in range(n_batches):
            s = b * batch_size
            e = min(s + batch_size, n_windows)

            # [B, L, K] → [B, K, L] sesuai format model
            bd = torch.tensor(data_w[s:e], dtype=torch.float32
                              ).permute(0,2,1).to(configs.device)
            bm = torch.tensor(obs_w[s:e], dtype=torch.float32
                              ).permute(0,2,1).to(configs.device)
            bt = torch.tensor(
                np.tile(np.arange(seq_len, dtype=np.float32), (e-s, 1))
            ).to(configs.device)   # [B, seq_len] — identik format DataLoader

            # Dapatkan side_info dulu
            side_info = model.get_side_info(bt, bm)

            # impute: ambil n_samples prediksi, ambil median
            samples = model.impute(bd, bm, side_info, n_samples=20)  # 20 cukup untuk output operasional
            # samples shape: [B, n_samples, K, L]
            imp_med = samples.median(dim=1).values   # [B, K, L]
            imp_med = imp_med.permute(0, 2, 1).cpu().numpy()  # [B, L, K]

            # Isi posisi missing dengan prediksi, observed tetap nilai asli
            miss_b = miss_w[s:e]                     # [B, L, K] bool
            result_w[s:e][miss_b] = imp_med[miss_b]

            if (b+1) % 10 == 0 or b == n_batches-1:
                print(f"  batch {b+1}/{n_batches} done")

    # ── Flatten dan denormalisasi ─────────────────────────────────────────────
    result_flat = result_w.reshape(n_used, K)          # [n_used, K]
    result_orig = result_flat * stds + means            # skala asli

    # ── Susun DataFrame numerik ───────────────────────────────────────────────
    num_cols = get_num_cols(dataset)
    if len(num_cols) == K:
        df_num = pd.DataFrame(result_orig, columns=num_cols)
    else:
        df_num = pd.DataFrame(result_orig)

    # ── Gabungkan dengan kategorik (jika raw CSV tersedia) ───────────────────
    if raw_csv and os.path.exists(raw_csv):
        df_raw  = pd.read_csv(raw_csv)
        cat_cols = [c for c in df_raw.columns if c not in (num_cols or df_raw.select_dtypes(exclude=['object']).columns.tolist())]
        df_cat  = df_raw[cat_cols].iloc[:n_used].reset_index(drop=True)
        df_out  = pd.concat([df_cat, df_num.reset_index(drop=True)], axis=1)
        # Susun kolom sesuai urutan asli
        orig_order = [c for c in df_raw.columns if c in df_out.columns]
        df_out = df_out[orig_order]
    else:
        df_out = df_num

    # ── Kolom audit ───────────────────────────────────────────────────────────
    n_train_w = round(n_windows * 0.7)
    n_train_r = n_train_w * seq_len
    df_out['_is_test'] = 0
    df_out.loc[n_train_r:, '_is_test'] = 1

    # ── Simpan ────────────────────────────────────────────────────────────────
    out_path = f'Data/{dataset.upper()}_HWD_full_mr{rate}.csv'
    df_out.to_csv(out_path, index=False)

    nan_left = df_out[df_out.columns.difference(['_is_test'])].isna().sum().sum()
    print(f"[full_inference] Saved: {out_path}  "
          f"shape={df_out.shape}  NaN={nan_left}")
    return df_out

# ══════════════════════════════════════════════════════════════════════════════
# Test / evaluasi
# ══════════════════════════════════════════════════════════════════════════════

def diffusion_test(configs, model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    all_target            = []
    all_evalpoint         = []
    all_generated_samples = []

    target_2d   = []
    forecast_2d = []
    eval_p_2d   = []
    generate_data2d = []

    start = time.time()
    print(f"Test batches: {len(test_loader)}")

    for i, (observed_data, observed_mask, observed_tp, gt_mask) in enumerate(test_loader):

        output = model.evaluate(
            observed_data, observed_mask, observed_tp, gt_mask,
            n_samples=configs.n_samples
        )
        imputed_samples, c_target, eval_points, observed_points, observed_time = output

        # [B, n_samples, K, L] → [B, n_samples, L, K]
        imputed_samples = imputed_samples.permute(0, 1, 3, 2)
        c_target        = c_target.permute(0, 2, 1)
        eval_points     = eval_points.permute(0, 2, 1)
        observed_points = observed_points.permute(0, 2, 1)

        all_target.append(c_target)
        all_evalpoint.append(eval_points)
        all_generated_samples.append(imputed_samples)

        # Deterministik: median sampel
        imputed_median = imputed_samples.median(dim=1).values.detach().cpu()
        # eval_points: hanya posisi artificial missing (dari mask eksperimen)
        # = posisi yang di mask eksperimen = 0 TAPI di gt_mask = 1
        # Ini mengecualikan natural missing & outlier dari evaluasi metrik
        evalmask       = (gt_mask - observed_mask).clamp(0, 1)
        imputed_data   = (observed_mask * observed_data
                          + (1 - observed_mask) * imputed_median)

        target_2d.append(observed_data)
        forecast_2d.append(imputed_data)
        eval_p_2d.append(evalmask)

        B, L, K = imputed_data.shape
        generate_data2d.append(
            imputed_data.reshape(B * L, K).detach().cpu().numpy()
        )
        print(f"  batch {i+1} | elapsed {time.time()-start:.1f}s")
        start = time.time()

    # ── Simpan hasil imputasi (test set, skala asli) ─────────────────────────
    generate_data2d = np.vstack(generate_data2d)    # [n_test_rows, K]
    out_csv = (f"HWD_Imputation_{configs.dataset}"
               f"_mr{configs.missing_rate}_original.csv")

    # Denormalisasi ke skala asli
    means_path = f'Data/{configs.dataset.upper()}_means.npy'
    stds_path  = f'Data/{configs.dataset.upper()}_stds.npy'
    if os.path.exists(means_path):
        means = np.load(means_path)
        stds  = np.load(stds_path)
        generate_data2d_orig = generate_data2d * stds + means
    else:
        generate_data2d_orig = generate_data2d   # fallback: simpan z-score

    # Simpan dengan pd.DataFrame agar header nama kolom benar
    num_cols = get_num_cols(configs.dataset)
    if len(num_cols) == generate_data2d_orig.shape[1]:
        df_out = pd.DataFrame(generate_data2d_orig, columns=num_cols)
    else:
        df_out = pd.DataFrame(generate_data2d_orig)
    df_out.to_csv(out_csv, index=False)
    print(f"Imputation saved: {out_csv}  "
          f"min={generate_data2d_orig.min():.3f}  "
          f"max={generate_data2d_orig.max():.3f}")

    # ── Metrik (di Z-score space, sama seperti FGTI) ─────────────────────────
    target_2d   = torch.cat(target_2d,   dim=0)
    forecast_2d = torch.cat(forecast_2d, dim=0)
    eval_p_2d   = torch.cat(eval_p_2d,   dim=0)

    MAE  = calc_MAE(target_2d,  forecast_2d, eval_p_2d)
    RMSE = calc_RMSE(target_2d, forecast_2d, eval_p_2d)

    all_target    = torch.cat(all_target,           dim=0)
    all_evalpoint = torch.cat(all_evalpoint,         dim=0)
    all_generated = torch.cat(all_generated_samples, dim=0)

    CRPS = utils.calc_quantile_CRPS(
        target=all_target,
        forecast=all_generated,
        eval_points=all_evalpoint,
        mean_scaler=0,
        scaler=1
    )

    print(f"  HWD  MAE={MAE:.4f}  RMSE={RMSE:.4f}  CRPS={CRPS:.4f}")

    # Full inference: imputasi semua 8016 baris × semua kolom dengan HWD
    full_inference_and_save(configs, model)

    return {"MAE": MAE.item(), "RMSE": RMSE.item(), "CRPS": CRPS}


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    return torch.sqrt(torch.mean((target[eval_p] - forecast[eval_p]) ** 2))


def calc_MAE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    return torch.mean(torch.abs(target[eval_p] - forecast[eval_p]))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    configs = get_config()

    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)

    # Preprocessing — skip otomatis kalau file sudah ada
    if configs.dataset == 'kdd':
        preprocess_kdd()
    elif configs.dataset == 'imk':
        preprocess_imk()
    elif configs.dataset in ('guangzhou', 'physio'):
        # Guangzhou dan PhysioNet sudah punya _norm.csv dari FGTI pipeline.
        # Kalau belum ada, buat dulu dengan cara yang sama seperti preprocess_kdd:
        norm_files = {'guangzhou': 'Data/Guangzhou_norm.csv',
                      'physio':    'Data/Physio_norm.csv'}
        raw_files  = {'guangzhou': 'Data/guangzhou.csv',
                      'physio':    'Data/physio.csv'}
        norm_path  = norm_files[configs.dataset]
        raw_path   = raw_files[configs.dataset]
        if not os.path.exists(norm_path) and os.path.exists(raw_path):
            print(f'[preprocess] Membuat {norm_path} dari {raw_path}...')
            preprocess_generic(raw_path, norm_path,
                               prefix=configs.dataset.capitalize())

    if configs.incremental and configs.base_ckpt:
        import incremental
        model = main_model.HWD(configs).to(configs.device)
        model.load_state_dict(
            torch.load(configs.base_ckpt, map_location=configs.device)
        )
        print(f"Loaded base checkpoint: {configs.base_ckpt}")
        train_loader, _ = A_dataset.get_dataset(configs)
        model = incremental.incremental_finetune(
            model, train_loader,
            freeze_ratio=configs.freeze_ratio,
            lr=configs.incr_lr,
            epochs=configs.incr_epochs
        )
    else:
        model = diffusion_train(configs)

    print("─── TEST ───")
    diffusion_test(configs, model)