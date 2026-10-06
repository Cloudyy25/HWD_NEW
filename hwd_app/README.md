# HWD Imputation Explorer

Aplikasi Streamlit untuk demonstrasi model **Hierarchical Wavelet Diffusion (HWD)**,
artefak komputasi skripsi Cloudya Qashwah Montolalu (Politeknik Statistik STIS, 2026).

## Isi aplikasi

|Halaman|Isi|
|-|-|
|Tentang Model|Cara kerja HWD, spesifikasi (Tabel 3.5), konfigurasi per dataset (Tabel 3.6)|
|Benchmark|MAE dan RMSE HWD dibandingkan Mean, CSDI, FGTI; tabel ablation study|
|Visualisasi Hasil|Grafik dan galat per variabel dari berkas keluaran notebook yang nyata|
|Rule Engine Susenas|Aturan lompatan, uji interaktif, dan statistik dari data asli|
|Imputasi Susenas|Isi missing alami, atau uji dengan missing buatan beserta metrik rupiah dan scatter|
|Demo Inferensi|Unggah CSV, sembunyikan nilai secara artifisial, isi dengan HWD atau pembanding|

## Sumber angka

* **Mean, CSDI, FGTI**: Yang et al. (2024), mekanisme MCAR. MAE dan RMSE dari Tabel 1,
CRPS CSDI dan FGTI dari Tabel 5 pada lampiran paper.
* **HWD**: dibaca dari `results/hwd\_results.csv`. Tingkat missing yang belum dijalankan
ditampilkan sebagai "–", tidak diisi angka perkiraan.

Aplikasi ini tidak membuat angka atau hasil imputasi rekaan di halaman mana pun.

## Struktur repo untuk deploy

Letakkan folder `hwd\_app/` di dalam repo HWD, sejajar dengan kode model:

```
HWD\_NEW/
├── models/            main\_model.py, diffusion.py, ts\_transformer.py, \_\_init\_\_.py
├── layers/            Diff\_layers.py, Embed.py, \_\_init\_\_.py
├── survey\_rules.py
└── hwd\_app/
    ├── app.py
    ├── hwd\_core.py
    ├── requirements.txt
    ├── results/hwd\_results.csv
    ├── assets/gambar\_3\_3\_arsitektur.png   (opsional)
    └── .streamlit/config.toml
```

Aplikasi mencari folder `models/` di folder `hwd\_app/`, folder induknya, atau di lokasi yang
ditunjuk variabel lingkungan `HWD\_REPO`.

**Penting untuk `.gitignore`.** Bila `.gitignore` repo memblokir `\*.csv` dan `\*.png`,
tambahkan pengecualian berikut agar berkas hasil dan gambar ikut ter-push:

```
!hwd\_app/results/hwd\_results.csv
!hwd\_app/assets/\*.png
```

## Deploy ke Streamlit Community Cloud

1. Push repo ke GitHub.
2. Buka share.streamlit.io, pilih **New app**, pilih repo, lalu isi **Main file path**
dengan `hwd\_app/app.py`.
3. Deploy. Dependensi dipasang dari `hwd\_app/requirements.txt` (PyTorch versi CPU).

Menjalankan secara lokal:

```bash
cd hwd\_app
pip install -r requirements.txt
streamlit run app.py
```

## Memperbarui hasil HWD

Tambahkan satu baris per eksperimen ke `results/hwd\_results.csv`, tanpa mengubah kode:

```
dataset,mekanisme,missing\_rate,mode,konfigurasi,MAE,RMSE,CRPS,catatan
Guangzhou Traffic,MCAR,10,benchmark,full,0.xxxx,0.xxxx,0.xxxx,"db2, 3 tingkat"
Susenas ,MCAR,10,operational,full,0.xxxx,0.xxxx,0.xxxx,Kaggle
```

* `dataset` harus persis: `KDD Cup 2018`, `Guangzhou Traffic`, `PhysioNet 2012`, atau `Susenas`.
* `konfigurasi`: `full`, `no\_wavelet`, atau `no\_crosslevel`.
* Halaman Benchmark membandingkan dengan FGTI hanya baris `mode = benchmark` dan `konfigurasi = full`.
* Berkas CSV juga bisa diunggah langsung dari halaman Benchmark tanpa commit.

## Berkas untuk halaman Visualisasi Hasil

Diunggah dari keluaran notebook (sel "Simpan + line chart"):

* `data\_{stem}\_norm.npy`, `imputed\_{stem}\_norm.npy`, `evalmask\_{stem}.npy` (wajib)
* `condmask\_{stem}.npy` (opsional, untuk menandai imputasi pada missing alami)
* `means.npy` dan `stds.npy` (opsional, untuk skala asli)

Alternatifnya, unggah `evaluasi\_{stem}\_norm.csv` atau `evaluasi\_{stem}\_asli.csv`.

## Demo inferensi dengan checkpoint

* Unggah checkpoint `.pt` hasil pelatihan (Daubechies-2, tiga tingkat).
* Urutan kolom CSV harus sama dengan saat pelatihan; jumlah kolom harus sama dengan K checkpoint.
* Unggah juga `means.npy` dan `stds.npy` dari praproses (misalnya `Data/KDD\_means.npy` atau
`susenas\_means\_operational.npy`). Tanpa keduanya, normalisasi memakai statistik data unggahan
dan hasilnya dapat berbeda dari pelatihan.
* Langkah difusi tetap T = 50 sesuai pelatihan. Inferensi di CPU lambat, jadi mulailah
dari sedikit window dan sedikit sampel.

Tanpa checkpoint, halaman ini menyediakan Mean imputation dan interpolasi linear sebagai
pembanding sederhana, dan hasilnya diberi label bukan HWD.

## Catatan kompatibilitas

`requirements.txt` membatasi `pandas<3`. Pada pandas 3, kolom teks bertipe `str`, sehingga
pengecekan `series.dtype == object` di fungsi `\_is\_filled` pada `survey\_rules.py` terlewati
dan sel kosong dianggap terisi. Aplikasi ini sudah memakai pengecekan yang aman untuk kedua
versi. Disarankan memperbarui `survey\_rules.py` dengan pengecekan yang sama:

```python
if pd.api.types.is\_object\_dtype(series) or pd.api.types.is\_string\_dtype(series):
```

## Halaman Imputasi Susenas

Halaman ini memakai **satu** checkpoint, yaitu skenario C uji lintas tahun (latih 2023–2024). Model HWD tidak khusus untuk satu mekanisme missing, sehingga checkpoint yang sama dipakai untuk kedua mode. Untuk demo mode "Uji dengan missing buatan", gunakan data Susenas **2025**. Data tahun itu tidak pernah dilihat model, sehingga metriknya jujur.

### Dua berkas yang dibutuhkan

1. **Checkpoint ringan** tanpa state optimizer, supaya ukurannya kecil:

```python
   import torch
   ck = torch.load('hwd\_susenas\_holdout\_C\_…\_e100.pt', map\_location='cpu', weights\_only=False)
   slim = {k: ck\[k] for k in ('model', 'wavelet', 'levels', 'ablation', 'arch\_v2', 'mechanism', 'mode') if k in ck}
   torch.save(slim, 'hwd\_susenas\_C.pt')
   ```

2. **Paket model** berisi urutan variabel, means, stds, dan batas 3×IQR dari tahun latih. Jalankan di notebook lintas tahun setelah sel 3:

```python
   import json
   tr = np.isin(YEARS, \[2023, 2024])
   fences, means, stds = fit\_preprocess(X\_all\[tr])
   paket = {'feat\_cols': FEAT\_COLS, 'means': means.tolist(), 'stds': stds.tolist(),
            'fences': {FEAT\_COLS\[j]: \[float(lo), float(hi)] for j, (lo, hi) in fences.items()},
            'seq\_len': SEQ\_LEN, 'mode': MODE, 'train\_years': \[2023, 2024]}
   json.dump(paket, open(f'{OUT\_DIR}/paket\_model\_susenas\_C.json', 'w'), indent=1)
   ```

Letakkan keduanya di `hwd\_app/model\_susenas/`. Aplikasi akan memakainya secara otomatis. Kalau folder itu kosong, halaman menampilkan tombol unggah untuk kedua berkas.

Karena `.gitignore` memblokir `\*.pt`, tambahkan pengecualian berikut. Pastikan juga ukuran checkpoint di bawah 100 MB, batas berkas GitHub.

```
!hwd\_app/model\_susenas/\*.pt
!hwd\_app/model\_susenas/\*.json
```

### Alur di aplikasi

* **Data:** CSV atau XLSX dengan kolom sesuai `feat\_cols` di paket model, tanpa membedakan huruf besar-kecil, ditambah R703\_A untuk rule engine. Jumlah baris bebas: baris sisa dilengkapi padding menjadi kelipatan 48, lalu padding dibuang setelah imputasi.
* **Isi missing alami:** sel kosong dan outlier 3×IQR tidak dipakai sebagai kondisi. Model mengisi sel kosong, lalu rule engine mengosongkan kembali structural missing. Nilai outlier asli **dipertahankan** kecuali opsi penggantian dinyalakan. Keluarannya berupa berkas CSV dengan kolom `\_diimputasi` dan `\_outlier`.
* **Uji dengan missing buatan:** sebagian nilai teramati disembunyikan (MCAR, MAR, atau block; 10–40%). Aplikasi menampilkan MAE, RMSE, dan CRPS dalam skala Z, tabel MAE, RMSE, dan MAPE per variabel pengeluaran dalam rupiah, scatter 2×2, serta dua berkas unduhan: data hasil imputasi dan daftar posisi uji.
* **Keluaran:** variabel kode dibulatkan ke kode sah yang ada di data (bisa dimatikan), dan pengeluaran hasil imputasi dibatasi minimal 0.

