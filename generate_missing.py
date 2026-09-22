"""
generate_missing.py — HWD
Menggabungkan 6 file generate_missing_position_*.py FGTI menjadi satu.

Jalankan SEKALI sebelum training untuk membuat file mask yang disimpan
ke disk. Training loop akan load mask ini via dataset.py.

Mekanisme missing yang didukung:
  - MCAR  : Missing Completely At Random — acak seragam
  - MAR   : Missing At Random — probabilitas missing bergantung variabel lain
  - MNAR  : Missing Not At Random — probabilitas missing bergantung nilai sendiri
  - BLOCK : Block consecutive missing — simulasi late response IMK

Cara pakai CLI:
  # Generate semua mask MCAR untuk KDD
  python generate_missing.py --dataset kdd --mechanism mcar

  # Generate block missing untuk IMK
  python generate_missing.py --dataset imk --mechanism block \
      --data_path Data/IMK_norm.csv --block_size 2

  # Generate semua sekaligus (untuk benchmark)
  python generate_missing.py --dataset all --mechanism all

Cara pakai dari notebook:
  import generate_missing as gm
  gm.generate_all_masks('kdd', mechanisms=['mcar','mar','mnar'])
  gm.generate_all_masks('imk', mechanisms=['mcar','block'],
                        data_path='Data/IMK_norm.csv')
"""

import os
import random
import argparse
import numpy as np


# ─────────────────────────────────────────────────────────
# Konfigurasi default — identik FGTI
# ─────────────────────────────────────────────────────────
MISSING_RATES = [0.1, 0.2, 0.3, 0.4]
SEEDS         = [3407, 3408, 3409, 3410, 3411]

DATASET_CONFIG = {
    "kdd": {
        "data_file" : "Data/KDD_norm.csv",
        "mask_dir"  : "Data/mask/kdd",
        "prefix"    : "kdd",
        # Kolom trigger untuk MAR — temperature station (identik FGTI)
        "mar_col"   : 6,
    },
    "guangzhou": {
        "data_file" : "Data/Guangzhou_norm.csv",
        "mask_dir"  : "Data/mask/guangzhou",
        "prefix"    : "guangzhou",
        "mar_col"   : 0,
    },
    "physio": {
        "data_file" : "Data/Physio_norm.csv",
        "mask_dir"  : "Data/mask/physio",
        "prefix"    : "physio",
        "mar_col"   : 0,
    },
    "imk": {
        "data_file" : "",          # diisi via --data_path
        "mask_dir"  : "Data/mask/imk",
        "prefix"    : "imk",
        "mar_col"   : 0,
    },
}


# ─────────────────────────────────────────────────────────
# Base mask — posisi yang sudah NaN/tidak valid dikecualikan
# ─────────────────────────────────────────────────────────
def get_base_mask(data: np.ndarray) -> np.ndarray:
    """
    mask_org: 1 = observed, 0 = sudah missing di data asli (-200 atau NaN).
    Missing artifisial HANYA dibangkitkan di posisi mask_org=1.
    """
    mask = np.ones_like(data, dtype=np.int32)
    mask[data == -200]     = 0
    mask[np.isnan(data)]   = 0
    return mask


# ─────────────────────────────────────────────────────────
# MCAR — identik logika FGTI, digabungkan
# ─────────────────────────────────────────────────────────
def generate_mcar(data: np.ndarray, missing_rate: float,
                  seed: int, valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Missing Completely At Random.
    valid_mask: posisi yang boleh di-missing (default = base_mask).
    Untuk IMK, pass valid_mask dari survey_rules.get_valid_mask().
    """
    random.seed(seed)
    np.random.seed(seed)

    base         = get_base_mask(data)
    if valid_mask is not None:
        base     = base * valid_mask.astype(np.int32)

    mask_target  = base.copy()
    target_count = int(np.sum(base) * missing_rate)
    valid_pos    = list(zip(*np.where(base == 1)))

    # Acak tanpa pengembalian — lebih efisien dari while loop FGTI
    chosen = np.random.choice(len(valid_pos), size=target_count, replace=False)
    for idx in chosen:
        i, j = valid_pos[idx]
        mask_target[i, j] = 0

    return mask_target


# ─────────────────────────────────────────────────────────
# MAR — identik logika FGTI
# ─────────────────────────────────────────────────────────

def _efraimidis(flat_prob: np.ndarray, size: int) -> np.ndarray:
    """
    Weighted sampling tanpa pengembalian pada flat index menggunakan
    Efraimidis-Spirakis (2006): key = -log(U) / p, ambil top-size terkecil.
    Kompleksitas O(N) memori + O(N log size) waktu via argpartition.
    Menggunakan float32 untuk efisiensi memori.
    """
    u    = np.random.uniform(size=len(flat_prob)).astype(np.float32)
    keys = -np.log(u) / (flat_prob.astype(np.float32) + np.float32(1e-10))
    return np.argpartition(keys, size)[:size]


def generate_mar(data: np.ndarray, missing_rate: float,
                 seed: int, mar_col: int = 0,
                 valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Missing At Random (MAR).
    P(missing | row i, col j) ∝ rank(row i berdasarkan mar_col).
    Kolom dipilih secara uniform sehingga:
        P(i, j) = (1/K) × prob_i
    Implementasi: flat-index Efraimidis tanpa argwhere (hemat memori).
    """
    np.random.seed(seed)

    base = get_base_mask(data)
    if valid_mask is not None:
        base = base * valid_mask.astype(np.int32)

    N, K = data.shape
    target_count = int(np.sum(base) * missing_rate)

    # Hitung probabilitas baris berdasarkan mar_col
    attr = data[:, mar_col].copy().astype(np.float64)
    attr[attr == -200] = np.nanmin(attr[attr != -200]) if (attr != -200).any() else 0.0
    attr = np.nan_to_num(attr, nan=float(np.nanmin(attr)) if not np.all(np.isnan(attr)) else 0.0)
    rank = (np.argsort(np.argsort(attr)) + 1).astype(np.float32)
    row_prob = rank / rank.sum()   # [N]

    # Flat prob: posisi (i,j) → index i*K+j, prob = row_prob[i] / K
    # (kolom uniform, baris berdasarkan MAR rank)
    flat_prob = np.repeat(row_prob, K) / K  # [N*K]

    # Nol-kan posisi yang tidak valid (base==0)
    flat_prob *= base.flatten().astype(np.float32)
    flat_prob /= flat_prob.sum()  # renormalisasi

    # Sample target_count posisi
    chosen_flat = _efraimidis(flat_prob, target_count)
    mask = base.copy()
    mask.flat[chosen_flat] = 0
    return mask


def generate_mnar(data: np.ndarray, missing_rate: float,
                  seed: int, valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Missing Not At Random (MNAR).
    P(missing | row i, col j) ∝ rank(nilai data[i,j] dalam kolom j).
    Nilai besar lebih mungkin hilang.
    Implementasi: flat-index Efraimidis tanpa argwhere (hemat memori).
    """
    np.random.seed(seed)

    base = get_base_mask(data)
    if valid_mask is not None:
        base = base * valid_mask.astype(np.int32)

    N, K = data.shape
    target_count = int(np.sum(base) * missing_rate)

    # Hitung col_probs [N, K]: prob tiap posisi berdasarkan rank dalam kolomnya
    col_probs = np.zeros((N, K), dtype=np.float32)
    for c in range(K):
        col_data = data[:, c].copy().astype(np.float64)
        valid = col_data != -200
        min_val = float(col_data[valid].min()) if valid.any() else 0.0
        col_data[~valid] = min_val
        col_data = np.nan_to_num(col_data, nan=min_val)
        rank = (np.argsort(np.argsort(col_data)) + 1).astype(np.float32)
        col_probs[:, c] = rank / rank.sum()

    # Flat prob: (i,j) → i*K+j, prob = col_probs[i,j] / K
    flat_prob = col_probs.flatten() / K

    # Nol-kan posisi tidak valid
    flat_prob *= base.flatten().astype(np.float32)
    flat_prob /= flat_prob.sum()

    chosen_flat = _efraimidis(flat_prob, target_count)
    mask = base.copy()
    mask.flat[chosen_flat] = 0
    return mask


def generate_block(data: np.ndarray, missing_rate: float,
                   seed: int, block_size: int = 2,
                   valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Block consecutive missing — mensimulasikan late response.
    Pilih ~missing_rate proporsi unit (baris window), lalu
    hilangkan blok block_size timestep berturutan per unit.

    data shape diasumsikan [N_windows * seq_len, D] — flat 2D.
    seq_len diinfer dari block_size dan proporsi.

    Untuk data panel IMK: satu 'unit' = satu window [seq_len, D].
    """
    random.seed(seed)
    np.random.seed(seed)

    base          = get_base_mask(data)
    if valid_mask is not None:
        base      = base * valid_mask.astype(np.int32)

    mask_target   = base.copy()
    n_rows, n_cols = data.shape

    # Perkiraan jumlah window yang perlu di-block
    # agar total missing ≈ missing_rate * total_observed
    total_obs     = np.sum(base)
    target_count  = int(total_obs * missing_rate)
    missing_count = 0

    # Vectorized: pre-generate semua kandidat blok sekaligus
    # Estimasi jumlah blok yang dibutuhkan (setiap blok isi block_size posisi)
    n_blocks_needed = (target_count // block_size) + 1
    # Oversample 3x untuk antisipasi collision/overlap
    n_candidates    = n_blocks_needed * 3

    row_starts = np.random.randint(0, max(1, n_rows - block_size + 1),
                                   size=n_candidates)
    cols       = np.random.randint(0, n_cols, size=n_candidates)

    for i in range(n_candidates):
        if missing_count >= target_count:
            break
        r0  = row_starts[i]
        col = cols[i]
        r1  = min(r0 + block_size, n_rows)
        # Hanya proses kalau ada posisi valid di blok ini
        if mask_target[r0:r1, col].sum() == 0:
            continue
        for r in range(r0, r1):
            if missing_count >= target_count:
                break
            if mask_target[r, col] == 1:
                mask_target[r, col] = 0
                missing_count += 1

    return mask_target


# ─────────────────────────────────────────────────────────
# Fungsi tingkat tinggi — generate dan simpan semua mask
# ─────────────────────────────────────────────────────────
def generate_all_masks(dataset: str,
                       mechanisms: list = None,
                       missing_rates: list = None,
                       seeds: list = None,
                       data_path: str = '',
                       block_size: int = 2,
                       valid_mask: np.ndarray = None):
    """
    Generate dan simpan semua file mask ke disk.

    Args:
        dataset      : 'kdd' | 'guangzhou' | 'physio' | 'imk'
        mechanisms   : list subset dari ['mcar','mar','mnar','block']
                       default: ['mcar','mar','mnar'] untuk benchmark,
                                ['mcar','block'] untuk imk
        missing_rates: list rate (default [0.1, 0.2, 0.3, 0.4])
        seeds        : list seed (default [3407..3411])
        data_path    : path ke data CSV — wajib diisi untuk 'imk'
        block_size   : panjang blok untuk mekanisme 'block'
        valid_mask   : [N, D] boolean — posisi valid untuk IMK
    """
    if mechanisms is None:
        mechanisms = ['mcar', 'block'] if dataset == 'imk' \
                     else ['mcar', 'mar', 'mnar']
    if missing_rates is None:
        missing_rates = MISSING_RATES
    if seeds is None:
        seeds = SEEDS

    cfg = DATASET_CONFIG[dataset].copy()
    if data_path:
        cfg['data_file'] = data_path

    # Load data
    print(f"Loading {cfg['data_file']} ...")
    if cfg['data_file'].endswith('.npy'):
        data = np.load(cfg['data_file']).astype(np.float64)
    else:
        data = np.loadtxt(cfg['data_file'], delimiter=',')
    print(f"  Shape: {data.shape}")

    # Buat direktori output
    os.makedirs(cfg['mask_dir'], exist_ok=True)

    total_files = 0
    for mechanism in mechanisms:
        for rate in missing_rates:
            for seed in seeds:
                # Tentukan nama file output
                if mechanism == 'mcar':
                    fname = f"{cfg['prefix']}_{rate}_{seed}.csv"
                elif mechanism == 'mar':
                    fname = f"{cfg['prefix']}mar_{rate}_{seed}.csv"
                elif mechanism == 'mnar':
                    fname = f"{cfg['prefix']}mnar_{rate}_{seed}.csv"
                elif mechanism == 'block':
                    fname = f"{cfg['prefix']}block{block_size}_{rate}_{seed}.csv"
                else:
                    continue

                out_path = os.path.join(cfg['mask_dir'], fname)

                # Skip jika sudah ada
                if os.path.exists(out_path):
                    print(f"  Skip (exists): {fname}")
                    continue

                # Generate mask
                if mechanism == 'mcar':
                    mask = generate_mcar(data, rate, seed, valid_mask)
                elif mechanism == 'mar':
                    mask = generate_mar(data, rate, seed,
                                        cfg['mar_col'], valid_mask)
                elif mechanism == 'mnar':
                    mask = generate_mnar(data, rate, seed, valid_mask)
                elif mechanism == 'block':
                    mask = generate_block(data, rate, seed,
                                          block_size, valid_mask)

                np.savetxt(out_path, mask, fmt='%d', delimiter=',')
                actual_rate = 1 - mask.sum() / get_base_mask(data).sum()
                print(f"  Saved: {fname}  "
                      f"(actual missing: {actual_rate:.3f})")
                total_files += 1

    print(f"\nDone. {total_files} file mask baru dibuat di {cfg['mask_dir']}/")


# ─────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HWD — generate missing masks')
    parser.add_argument('--dataset',    type=str, default='kdd',
                        help='kdd | guangzhou | physio | imk | all')
    parser.add_argument('--mechanism',  type=str, default='mcar',
                        help='mcar | mar | mnar | block | all')
    parser.add_argument('--data_path',  type=str, default='',
                        help='Path ke data CSV (wajib untuk --dataset imk)')
    parser.add_argument('--block_size', type=int, default=2,
                        help='Panjang blok untuk mekanisme block (default 2)')
    parser.add_argument('--missing_rates', type=float, nargs='+',
                        default=MISSING_RATES)
    parser.add_argument('--seeds',     type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()

    # Expand 'all'
    datasets   = list(DATASET_CONFIG.keys()) \
                 if args.dataset == 'all' else [args.dataset]
    mechanisms = ['mcar', 'mar', 'mnar', 'block'] \
                 if args.mechanism == 'all' else [args.mechanism]

    for ds in datasets:
        print(f"\n{'═'*50}")
        print(f"Dataset: {ds.upper()}  |  Mechanisms: {mechanisms}")
        print(f"{'═'*50}")

        # Untuk IMK, block dan mcar saja secara default
        mech = mechanisms
        if ds == 'imk' and args.mechanism == 'all':
            mech = ['mcar', 'block']

        generate_all_masks(
            dataset       = ds,
            mechanisms    = mech,
            missing_rates = args.missing_rates,
            seeds         = args.seeds,
            data_path     = args.data_path if ds == 'imk' else '',
            block_size    = args.block_size,
        )