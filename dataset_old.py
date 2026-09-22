"""
dataset.py — HWD
================
Perubahan dari FGTI:
  - Hapus seluruh blok FFT (dataf, maxdataf, torchcde)
  - __getitem__ return 4 item (hapus dataf_res)
  - Tambah IMK_DATASET untuk data panel IMK BPS
  - Tambah GENERIC_DATASET untuk dataset baru tanpa ubah kode
  - get_dataset() tambah case "imk" dan fallback generic
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ══════════════════════════════════════════════════════════════════════════════
# KDD_DATASET
# ══════════════════════════════════════════════════════════════════════════════

class KDD_DATASET(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # ── Load data ─────────────────────────────────────────────────────────
        raw  = np.loadtxt("Data/KDD_norm.csv", delimiter=",")
        n_ok = (len(raw) // configs.seq_len) * configs.seq_len
        self.data = raw[:n_ok].reshape(-1, configs.seq_len, configs.enc_in)

        # ── Load atau buat mask ───────────────────────────────────────────────
        if configs.missing_rate == 0:
            # Tidak ada artificial missing — hanya natural missing
            self.mask = np.ones_like(self.data)
            self.mask[self.data == -200] = 0
        else:
            mask_path = (f"Data/mask/kdd/kdd_{configs.missing_rate}"
                         f"_{configs.seed}.csv")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask file tidak ditemukan: {mask_path}\n"
                    f"Jalankan generate_missing.py terlebih dahulu."
                )
            mask_raw = np.loadtxt(mask_path, delimiter=",")
            self.mask = mask_raw[:n_ok].reshape(-1, configs.seq_len, configs.enc_in)

        # ── Ground-truth mask: semua posisi yang bukan -200 ──────────────────
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[self.data == -200] = 0

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index]).float(),
            torch.from_numpy(self.mask[index]).float(),
            torch.from_numpy(np.arange(self.configs.seq_len, dtype=np.float32)),
            torch.from_numpy(self.mask_gt[index]).float(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# GUANGZHOU_DATASET
# ══════════════════════════════════════════════════════════════════════════════

class GUANGZHOU_DATASET(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        raw  = np.loadtxt("Data/Guangzhou_norm.csv", delimiter=",")
        n_ok = (len(raw) // configs.seq_len) * configs.seq_len
        self.data = raw[:n_ok].reshape(-1, configs.seq_len, configs.enc_in)

        mask_path = (f"Data/mask/guangzhou/guangzhou_{configs.missing_rate}"
                     f"_{configs.seed}.csv")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask file tidak ditemukan: {mask_path}\n"
                f"Jalankan generate_missing.py terlebih dahulu."
            )
        mask_raw  = np.loadtxt(mask_path, delimiter=",")
        self.mask = mask_raw[:n_ok].reshape(-1, configs.seq_len, configs.enc_in)

        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[self.data == -200] = 0

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index]).float(),
            torch.from_numpy(self.mask[index]).float(),
            torch.from_numpy(np.arange(self.configs.seq_len, dtype=np.float32)),
            torch.from_numpy(self.mask_gt[index]).float(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHYSIO_DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PHYSIO_DATASET(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        raw  = np.loadtxt("Data/Physio_norm.csv", delimiter=",")
        n_ok = (len(raw) // configs.seq_len) * configs.seq_len
        self.data = raw[:n_ok].reshape(-1, configs.seq_len, configs.enc_in)

        # ── Load mask eksperimen (identik KDD) ────────────────────────────────
        if configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[self.data == -200] = 0
        else:
            mask_path = (f"Data/mask/physio/physio_{configs.missing_rate}"
                         f"_{configs.seed}.csv")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask file tidak ditemukan: {mask_path}\n"
                    f"Jalankan generate_missing.py terlebih dahulu."
                )
            mask_raw  = np.loadtxt(mask_path, delimiter=",")
            self.mask = mask_raw[:n_ok].reshape(-1, configs.seq_len, configs.enc_in)

        # ── Ground-truth mask: semua posisi yang bukan -200 ──────────────────
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[self.data == -200] = 0

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index]).float(),
            torch.from_numpy(self.mask[index]).float(),
            torch.from_numpy(np.arange(self.configs.seq_len, dtype=np.float32)),
            torch.from_numpy(self.mask_gt[index]).float(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# IMK_DATASET — data panel IMK BPS
# ══════════════════════════════════════════════════════════════════════════════

class IMK_DATASET(Dataset):
    """
    Dataset untuk data panel IMK BPS.

    configs yang dibutuhkan (selain yang standar):
      configs.data_path      — path ke IMK_norm.csv (wajib)
      configs.col_names_path — path ke IMK_raw.csv dengan nama kolom
                               (opsional, untuk survey rule engine)
      configs.exp_mask_path  — path ke mask eksperimen MCAR/block
                               (opsional, kalau kosong pakai mask_gt)
    """
    def __init__(self, configs, split='train'):
        super().__init__()
        self.configs = configs
        self.split   = split

        # ── Load data ─────────────────────────────────────────────────────────
        data_path = getattr(configs, 'data_path', 'Data/IMK_norm.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"IMK data tidak ditemukan: {data_path}")

        raw = (np.load(data_path) if data_path.endswith('.npy')
               else np.loadtxt(data_path, delimiter=',')).astype(np.float32)

        # ── Survey Rule Engine: identifikasi structural missing ───────────────
        col_names_path = getattr(configs, 'col_names_path', '')
        if col_names_path and os.path.exists(col_names_path):
            import survey_rules
            df_raw         = pd.read_csv(col_names_path)
            valid_mask_2d  = survey_rules.get_valid_mask(df_raw)   # [N, D] bool
        else:
            valid_mask_2d  = np.ones(raw.shape, dtype=bool)

        # ── Mask ground-truth: observed DAN bukan structural ─────────────────
        mask_raw              = np.ones_like(raw)
        mask_raw[raw == -200] = 0
        mask_raw[np.isnan(raw)] = 0
        mask_raw              = mask_raw * valid_mask_2d.astype(np.float32)

        self.structural_mask  = (~valid_mask_2d).astype(np.float32)

        # ── Reshape ke windows ────────────────────────────────────────────────
        n_rows, D  = raw.shape
        n_windows  = n_rows // configs.seq_len
        n_ok       = n_windows * configs.seq_len

        data_w = raw[:n_ok].reshape(n_windows, configs.seq_len, D)
        mask_w = mask_raw[:n_ok].reshape(n_windows, configs.seq_len, D)
        stru_w = self.structural_mask[:n_ok].reshape(n_windows, configs.seq_len, D)

        # ── Train/test split 70/30 ────────────────────────────────────────────
        n_train = int(n_windows * 0.7)
        sl      = slice(None, n_train) if split == 'train' else slice(n_train, None)

        self.data          = data_w[sl]
        self.mask_gt       = mask_w[sl]
        self.structural_3d = stru_w[sl]

        # ── Mask eksperimen dari generate_missing.py ──────────────────────────
        exp_mask_path = getattr(configs, 'exp_mask_path', '')
        if exp_mask_path and os.path.exists(exp_mask_path):
            exp = (np.load(exp_mask_path) if exp_mask_path.endswith('.npy')
                   else np.loadtxt(exp_mask_path, delimiter=',')).astype(np.float32)
            exp_w     = exp[:n_ok].reshape(n_windows, configs.seq_len, D)
            self.mask = exp_w[sl]
        else:
            self.mask = self.mask_gt.copy()

        self.enc_in = D

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index]).float(),
            torch.from_numpy(self.mask[index]).float(),
            torch.from_numpy(np.arange(self.configs.seq_len, dtype=np.float32)),
            torch.from_numpy(self.mask_gt[index]).float(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC_DATASET — dataset baru tanpa ubah kode
# ══════════════════════════════════════════════════════════════════════════════

class GENERIC_DATASET(Dataset):
    """
    Untuk dataset selain KDD/Guangzhou/PhysioNet/IMK.
    Syarat: set configs.data_path ke file CSV yang sudah dinormalisasi.
    """
    def __init__(self, configs, split='train'):
        super().__init__()

        data_path = getattr(configs, 'data_path', '')
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(
                f"data_path tidak ditemukan: '{data_path}'\n"
                f"Set configs.data_path ke path CSV yang sudah dinormalisasi."
            )

        raw = (np.load(data_path) if data_path.endswith('.npy')
               else np.loadtxt(data_path, delimiter=',')).astype(np.float32)

        mask_path = getattr(configs, 'mask_path', '')
        if mask_path and os.path.exists(mask_path):
            mask_raw = (np.load(mask_path) if mask_path.endswith('.npy')
                        else np.loadtxt(mask_path, delimiter=',')).astype(np.float32)
        else:
            mask_raw              = np.ones_like(raw)
            mask_raw[raw == -200] = 0
            mask_raw[np.isnan(raw)] = 0

        n_rows, D  = raw.shape
        n_windows  = n_rows // configs.seq_len
        n_ok       = n_windows * configs.seq_len

        data_w = raw[:n_ok].reshape(n_windows, configs.seq_len, D)
        mask_w = mask_raw[:n_ok].reshape(n_windows, configs.seq_len, D)

        n_train    = int(n_windows * 0.7)
        sl         = slice(None, n_train) if split == 'train' else slice(n_train, None)
        self.data    = data_w[sl]
        self.mask_gt = mask_w[sl]

        exp_mask_path = getattr(configs, 'exp_mask_path', '')
        if exp_mask_path and os.path.exists(exp_mask_path):
            exp   = (np.load(exp_mask_path) if exp_mask_path.endswith('.npy')
                     else np.loadtxt(exp_mask_path, delimiter=',')).astype(np.float32)
            exp_w = exp[:n_ok].reshape(n_windows, configs.seq_len, D)
            self.mask = exp_w[sl]
        else:
            self.mask = self.mask_gt.copy()

        self.enc_in = D

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data[index]).float(),
            torch.from_numpy(self.mask[index]).float(),
            torch.from_numpy(np.arange(self.configs.seq_len, dtype=np.float32)),
            torch.from_numpy(self.mask_gt[index]).float(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_loaders(train_ds, test_ds, configs):
    """Helper: buat train + test DataLoader dari dua Dataset."""
    train_loader = DataLoader(
        train_ds, batch_size=configs.batch,
        num_workers=0, shuffle=True
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=configs.batch,
        num_workers=0, shuffle=False
    )
    return train_loader, test_loader


def get_kdd_dataset(configs):
    full     = KDD_DATASET(configs)
    n        = len(full)
    n_train  = int(n * 0.7)
    return _make_loaders(
        torch.utils.data.Subset(full, range(n_train)),
        torch.utils.data.Subset(full, range(n_train, n)),
        configs
    )


def get_guangzhou_dataset(configs):
    full    = GUANGZHOU_DATASET(configs)
    n       = len(full)
    n_train = int(n * 0.7)
    return _make_loaders(
        torch.utils.data.Subset(full, range(n_train)),
        torch.utils.data.Subset(full, range(n_train, n)),
        configs
    )


def get_physio_dataset(configs):
    full    = PHYSIO_DATASET(configs)
    n       = len(full)
    n_train = int(n * 0.7)
    return _make_loaders(
        torch.utils.data.Subset(full, range(n_train)),
        torch.utils.data.Subset(full, range(n_train, n)),
        configs
    )


def get_imk_dataset(configs):
    train_ds       = IMK_DATASET(configs, split='train')
    test_ds        = IMK_DATASET(configs, split='test')
    configs.enc_in = train_ds.enc_in
    configs.c_out  = train_ds.enc_in
    return _make_loaders(train_ds, test_ds, configs)


def get_generic_dataset(configs):
    train_ds       = GENERIC_DATASET(configs, split='train')
    test_ds        = GENERIC_DATASET(configs, split='test')
    configs.enc_in = train_ds.enc_in
    configs.c_out  = train_ds.enc_in
    return _make_loaders(train_ds, test_ds, configs)


def get_dataset(configs):
    """
    Entry point utama — dipanggil dari train.py.

    Dataset bawaan  : kdd | guangzhou | physio | imk
    Dataset baru    : set configs.dataset ke nama apapun
                      + configs.data_path ke CSV yang sudah dinormalisasi.
    """
    name = configs.dataset.lower()
    dispatch = {
        'kdd':       get_kdd_dataset,
        'guangzhou': get_guangzhou_dataset,
        'physio':    get_physio_dataset,
        'imk':       get_imk_dataset,
    }
    fn = dispatch.get(name, get_generic_dataset)
    return fn(configs)