# HWD Imputation Explorer

Aplikasi Streamlit untuk demonstrasi model **Hierarchical Wavelet Diffusion (HWD)**,
artefak komputasi skripsi Cloudya Qashwah Montolalu (Politeknik Statistik STIS, 2026).

## Isi aplikasi

| Halaman | Isi |
|---|---|
| Tentang Model | Cara kerja HWD, spesifikasi (Tabel 3.5), konfigurasi per dataset (Tabel 3.6) |
| Benchmark | MAE dan RMSE HWD dibandingkan Mean, CSDI, FGTI; tabel ablation study |
| Visualisasi Hasil | Grafik dan galat per variabel dari berkas keluaran notebook yang nyata |
| Rule Engine Susenas | Aturan lompatan, uji interaktif, dan statistik dari data asli |
| Imputasi Susenas | Isi missing alami, atau uji dengan missing buatan beserta metrik rupiah dan scatter |
| Demo Inferensi | Unggah CSV, sembunyikan nilai secara artifisial, isi dengan HWD atau pembanding |

## Sumber angka

- **Mean, CSDI, FGTI**: Yang et al. (2024), mekanisme MCAR. MAE dan RMSE dari Tabel 1,
  CRPS CSDI dan FGTI dari Tabel 5 pada lampiran paper.
- **HWD**: dibaca dari `results/hwd_results.csv`. Tingkat missing yang belum dijalankan
  ditampilkan sebagai "–", tidak diisi angka perkiraan.

Aplikasi ini tidak membuat angka atau hasil imputasi rekaan di halaman mana pun.

## Struktur repo untuk deploy

Letakkan folder `hwd_app/` di dalam repo HWD, sejajar dengan kode model:

```
HWD_NEW/
├── models/            main_model.py, diffusion.py, ts_transformer.py, __init__.py
├── layers/            Diff_layers.py, Embed.py, __init__.py
├── survey_rules.py
└── hwd_app/
    ├── app.py
    ├── hwd_core.py
    ├── requirements.txt
    ├── results/hwd_results.csv
    ├── assets/gambar_3_3_arsitektur.png   (opsional)
    └── .streamlit/config.toml
```

Aplikasi mencari folder `models/` di folder `hwd_app/`, folder induknya, atau di lokasi yang
ditunjuk variabel lingkungan `HWD_REPO`.

**Penting untuk `.gitignore`.** Bila `.gitignore` repo memblokir `*.csv` dan `*.png`,
tambahkan pengecualian berikut agar berkas hasil dan gambar ikut ter-push:

```
!hwd_app/results/hwd_results.csv
!hwd_app/assets/*.png
```

## Deploy ke Streamlit Community Cloud

1. Push repo ke GitHub.
2. Buka share.streamlit.io, pilih **New app**, pilih repo, lalu isi **Main file path**
   dengan `hwd_app/app.py`.
3. Deploy. Dependensi dipasang dari `hwd_app/requirements.txt` (PyTorch versi CPU).

Menjalankan secara lokal:

```bash
cd hwd_app
pip install -r requirements.txt
streamlit run app.py
```

## Memperbarui hasil HWD

Tambahkan satu baris per eksperimen ke `results/hwd_results.csv`, tanpa mengubah kode:

```
dataset,mekanisme,missing_rate,mode,konfigurasi,MAE,RMSE,CRPS,catatan
Guangzhou Traffic,MCAR,10,benchmark,full,0.xxxx,0.xxxx,0.xxxx,"db2, 3 tingkat"
Susenas KP,MCAR,10,operational,full,0.xxxx,0.xxxx,0.xxxx,Kaggle
```

- `dataset` harus persis: `KDD Cup 2018`, `Guangzhou Traffic`, `PhysioNet 2012`, atau `Susenas KP`.
- `konfigurasi`: `full`, `no_wavelet`, atau `no_crosslevel`.
- Halaman Benchmark membandingkan dengan FGTI hanya baris `mode = benchmark` dan `konfigurasi = full`.
- Berkas CSV juga bisa diunggah langsung dari halaman Benchmark tanpa commit.

## Berkas untuk halaman Visualisasi Hasil

Diunggah dari keluaran notebook (sel "Simpan + line chart"):

- `data_{stem}_norm.npy`, `imputed_{stem}_norm.npy`, `evalmask_{stem}.npy` (wajib)
- `condmask_{stem}.npy` (opsional, untuk menandai imputasi pada missing alami)
- `means.npy` dan `stds.npy` (opsional, untuk skala asli)

Alternatifnya, unggah `evaluasi_{stem}_norm.csv` atau `evaluasi_{stem}_asli.csv`.

## Demo inferensi dengan checkpoint

- Unggah checkpoint `.pt` hasil pelatihan (Daubechies-2, tiga tingkat).
- Urutan kolom CSV harus sama dengan saat pelatihan; jumlah kolom harus sama dengan K checkpoint.
- Unggah juga `means.npy` dan `stds.npy` dari praproses (misalnya `Data/KDD_means.npy` atau
  `susenas_means_operational.npy`). Tanpa keduanya, normalisasi memakai statistik data unggahan
  dan hasilnya dapat berbeda dari pelatihan.
- Langkah difusi tetap T = 50 sesuai pelatihan. Inferensi di CPU lambat, jadi mulailah
  dari sedikit window dan sedikit sampel.

Tanpa checkpoint, halaman ini menyediakan Mean imputation dan interpolasi linear sebagai
pembanding sederhana, dan hasilnya diberi label bukan HWD.

## Catatan kompatibilitas

`requirements.txt` membatasi `pandas<3`. Pada pandas 3, kolom teks bertipe `str`, sehingga
pengecekan `series.dtype == object` di fungsi `_is_filled` pada `survey_rules.py` terlewati
dan sel kosong dianggap terisi. Aplikasi ini sudah memakai pengecekan yang aman untuk kedua
versi. Disarankan memperbarui `survey_rules.py` dengan pengecekan yang sama:

```python
if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
```

## Halaman Imputasi Susenas

Halaman ini memakai **satu** checkpoint, yaitu skenario C uji lintas tahun (latih 2023–2024). Model HWD tidak khusus untuk satu mekanisme missing, sehingga checkpoint yang sama dipakai untuk kedua mode. Untuk demo mode "Uji dengan missing buatan", gunakan data Susenas **2025**. Data tahun itu tidak pernah dilihat model, sehingga metriknya jujur.

### Dua berkas yang dibutuhkan

1. **Checkpoint ringan** tanpa state optimizer, supaya ukurannya kecil:

   ```python
   import torch
   ck = torch.load('hwd_susenas_holdout_C_…_e100.pt', map_location='cpu', weights_only=False)
   slim = {k: ck[k] for k in ('model', 'wavelet', 'levels', 'ablation', 'arch_v2', 'mechanism', 'mode') if k in ck}
   torch.save(slim, 'hwd_susenas_C.pt')
   ```

2. **Paket model** berisi urutan variabel, means, stds, dan batas 3×IQR dari tahun latih. Jalankan di notebook lintas tahun setelah sel 3:

   ```python
   import json
   tr = np.isin(YEARS, [2023, 2024])
   fences, means, stds = fit_preprocess(X_all[tr])
   paket = {'feat_cols': FEAT_COLS, 'means': means.tolist(), 'stds': stds.tolist(),
            'fences': {FEAT_COLS[j]: [float(lo), float(hi)] for j, (lo, hi) in fences.items()},
            'seq_len': SEQ_LEN, 'mode': MODE, 'train_years': [2023, 2024]}
   json.dump(paket, open(f'{OUT_DIR}/paket_model_susenas_C.json', 'w'), indent=1)
   ```

Letakkan keduanya di `hwd_app/model_susenas/`. Aplikasi akan memakainya secara otomatis. Kalau folder itu kosong, halaman menampilkan tombol unggah untuk kedua berkas.

Karena `.gitignore` memblokir `*.pt`, tambahkan pengecualian berikut. Pastikan juga ukuran checkpoint di bawah 100 MB, batas berkas GitHub.

```
!hwd_app/model_susenas/*.pt
!hwd_app/model_susenas/*.json
```

### Alur di aplikasi

- **Data:** CSV atau XLSX dengan kolom sesuai `feat_cols` di paket model, tanpa membedakan huruf besar-kecil, ditambah R703_A untuk rule engine. Jumlah baris bebas: baris sisa dilengkapi padding menjadi kelipatan 48, lalu padding dibuang setelah imputasi.
- **Isi missing alami:** sel kosong dan outlier 3×IQR tidak dipakai sebagai kondisi. Model mengisi sel kosong, lalu rule engine mengosongkan kembali structural missing. Nilai outlier asli **dipertahankan** kecuali opsi penggantian dinyalakan. Keluarannya berupa berkas CSV dengan kolom `_diimputasi` dan `_outlier`.
- **Uji dengan missing buatan:** sebagian nilai teramati disembunyikan (MCAR, MAR, atau block; 10–40%). Aplikasi menampilkan MAE, RMSE, dan CRPS dalam skala Z, tabel MAE, RMSE, dan MAPE per variabel pengeluaran dalam rupiah, scatter 2×2, serta dua berkas unduhan: data hasil imputasi dan daftar posisi uji.
- **Keluaran:** variabel kode dibulatkan ke kode sah yang ada di data (bisa dimatikan), dan pengeluaran hasil imputasi dibatasi minimal 0.
