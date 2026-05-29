"""
survey_rules.py — HWD
File baru — tidak ada di FGTI maupun CSDI.

Tujuan:
  Layer 1 (pre-processing) dan Layer 3 (post-processing) untuk
  memisahkan M_structural (skip rules kuesioner) dari M_item
  (item nonresponse yang menjadi target imputasi).

  Hanya dipanggil saat --dataset imk.
  Untuk dataset lain (KDD, Guangzhou, PhysioNet) file ini tidak
  disentuh sama sekali — model tetap generic.

Cara kerja:
  1. get_valid_mask(df)      → valid_mask [N, D] boolean
     True  = posisi boleh diimputasi (M_item atau observed)
     False = M_structural (skip rules) — tidak diimputasi

  2. restore_structural(df_imputed, df_original) → DataFrame
     Kembalikan posisi structural missing ke NaN setelah imputasi.

Skip rules berdasarkan kuesioner VIMK24-S2 BPS.
Tambah/edit rules di IMK_SKIP_RULES untuk survei lain.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


# ─────────────────────────────────────────────────────────
# Skip rules VIMK24-S2
# Format setiap rule:
#   if_col  : nama kolom yang jadi trigger
#   if_val  : nilai trigger (atau list nilai)
#   null_cols: list kolom yang jadi structural missing jika trigger terpenuhi
#
# Tambahkan rule baru sesuai buku pedoman BPS jika diperlukan.
# ─────────────────────────────────────────────────────────
IMK_SKIP_RULES: List[Dict] = [
    # r203 = 2 (tidak aktif berproduksi) → Blok IV, V, VI, VII, VIII = structural
    {
        "if_col"   : "r203",
        "if_val"   : 2,
        "null_cols": [
            # Blok IV — pekerja dan balas jasa
            "r401_pekerja_dibayar", "r401_pekerja_tidak_dibayar",
            "r401_hari_kerja", "r401_jam_kerja",
            "r402_laki", "r402_perempuan",
            "r403_upah", "r403_iuran", "r403_lainnya", "r403_jumlah",
            # Blok V — pendapatan
            "r501_nilai_produksi", "r502_pendapatan_lain", "r503_jumlah",
            # Blok VI — biaya
            "r601_bahan_baku", "r602_pengeluaran_umum", "r603_non_operasional",
            # Blok VII — neraca
            "r701_aset", "r702_hutang", "r703_modal",
            # Blok VIII — ringkasan nilai tambah
            "r801_pendapatan", "r802_biaya", "r803_selisih",
        ]
    },
    # r207b != 5 (berbadan hukum) → r207c tidak relevan
    {
        "if_col"   : "r207b",
        "if_not"   : 5,
        "null_cols": [
            "r207c_pemisahan_pendapatan",
            "r207c_pemisahan_pengeluaran",
            "r207c_pemisahan_aset",
            "r207c_pemisahan_tabungan",
        ]
    },
    # r306a semua = 2 (tidak ada kemitraan) → r306b tidak relevan
    {
        "if_col"      : "r306a_kemitraan_any",
        "if_val"      : 0,   # 0 = tidak ada satupun kemitraan
        "null_cols"   : [
            "r306b_inti_plasma", "r306b_subkontrak",
            "r306b_perdagangan", "r306b_bagi_hasil",
            "r306b_kerja_sama", "r306b_joint_venture",
        ]
    },
]


# ─────────────────────────────────────────────────────────
# Fungsi utama
# ─────────────────────────────────────────────────────────

def get_valid_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Evaluasi skip rules dan return valid_mask.

    Args:
        df : DataFrame dengan kolom sesuai kuesioner IMK

    Returns:
        valid_mask : np.ndarray bool [N_rows, D_cols]
                     True  = posisi valid (observed atau M_item)
                     False = M_structural — jangan diimputasi
    """
    valid = np.ones((len(df), len(df.columns)), dtype=bool)

    col_idx = {col: i for i, col in enumerate(df.columns)}

    for rule in IMK_SKIP_RULES:
        trigger_col = rule["if_col"]

        # Skip rule jika kolom trigger tidak ada di data
        if trigger_col not in col_idx:
            continue

        # Tentukan baris yang terkena rule
        if "if_val" in rule:
            vals = rule["if_val"]
            if not isinstance(vals, list):
                vals = [vals]
            trigger_rows = df[trigger_col].isin(vals).values
        elif "if_not" in rule:
            trigger_rows = (df[trigger_col] != rule["if_not"]).values
        else:
            continue

        # Set False (structural) untuk kolom yang di-null
        for col in rule["null_cols"]:
            if col in col_idx:
                valid[trigger_rows, col_idx[col]] = False

    return valid   # [N, D]


def get_structural_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Kebalikan dari get_valid_mask.
    Return: structural_mask [N, D] — True = M_structural
    """
    return ~get_valid_mask(df)


def restore_structural(df_imputed: pd.DataFrame,
                       df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 3 post-processing:
    Kembalikan posisi structural missing ke NaN setelah imputasi.

    Args:
        df_imputed  : DataFrame hasil imputasi HWD
        df_original : DataFrame asli (sebelum imputasi)

    Returns:
        df_imputed dengan M_structural dikembalikan ke NaN
    """
    structural = get_structural_mask(df_original)  # [N, D]
    df_out     = df_imputed.copy()

    for j, col in enumerate(df_imputed.columns):
        df_out.loc[structural[:, j], col] = np.nan

    return df_out


def apply_valid_mask_to_numpy(data: np.ndarray,
                              df_original: pd.DataFrame) -> np.ndarray:
    """
    Helper: apply valid_mask ke numpy array hasil imputasi.
    Posisi structural missing dikembalikan ke np.nan.

    Args:
        data        : [N, D] numpy array hasil imputasi
        df_original : DataFrame asli untuk referensi skip rules

    Returns:
        data dengan M_structural = np.nan
    """
    structural = get_structural_mask(df_original)  # [N, D] bool
    result     = data.copy().astype(float)
    result[structural] = np.nan
    return result


# ─────────────────────────────────────────────────────────
# Quick test (jalankan: python survey_rules.py)
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Buat DataFrame dummy kecil untuk verifikasi
    dummy = pd.DataFrame({
        "r203"                     : [1, 2, 1, 2],   # 2 = tidak aktif
        "r207b"                    : [5, 3, 5, 1],   # 5 = tidak berbadan hukum
        "r207c_pemisahan_pendapatan": [1, 1, 1, 1],
        "r306a_kemitraan_any"      : [1, 0, 1, 0],
        "r306b_subkontrak"         : [1, 1, 1, 1],
        "r401_pekerja_dibayar"     : [5, 3, 2, 4],
        "r501_nilai_produksi"      : [1e6, 2e6, 3e6, 4e6],
    })

    valid = get_valid_mask(dummy)
    print("valid_mask:")
    print(pd.DataFrame(valid, columns=dummy.columns).to_string())

    # Row 1 dan 3 (r203=2) → kolom produksi dan pekerja harus False
    assert valid[1, dummy.columns.get_loc("r501_nilai_produksi")] == False
    assert valid[3, dummy.columns.get_loc("r501_nilai_produksi")] == False
    # Row 0 dan 2 (r203=1) → harus True
    assert valid[0, dummy.columns.get_loc("r501_nilai_produksi")] == True

    print("\nSemua assertion passed.")