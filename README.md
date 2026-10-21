# Hierarchical Wavelet Diffusion (HWD)

Repositori kode penelitian skripsi **Cloudya Qashwah Montolalu** mengenai *Hierarchical Wavelet Diffusion* (HWD) untuk imputasi data hilang pada data survei. HWD menggabungkan *conditional diffusion*, dekomposisi wavelet multilevel, dan *cross-level attention*. Penelitian mengevaluasi model pada tiga dataset benchmark dan data rumah tangga Susenas, serta menerapkan aturan kuesioner untuk mempertahankan sel *structural missing* tetap kosong.

**Penulis:** Cloudya Qashwah Montolalu  
**Kontak:** [cloudyamontolalu4@gmail.com](mailto:cloudyamontolalu4@gmail.com)  
**Dosen pembimbing:** Prof. Setia Pramana, S.Si., M.Sc., Ph.D.

## Tautan

| Sumber daya | Tautan |
|---|---|
| Aplikasi web HWD | [hwd-new.streamlit.app](https://hwd-new.streamlit.app/) |
| Repositori kode | [github.com/Cloudyy25/HWD_NEW](https://github.com/Cloudyy25/HWD_NEW) |

## Metode secara ringkas

HWD menggunakan proses difusi bersyarat untuk mengestimasi nilai yang hilang. *Discrete Wavelet Transform* (DWT) dengan wavelet Daubechies-2 membentuk representasi multilevel, yang kemudian digunakan sebagai kondisi model. *Cross-level attention* membantu menghubungkan informasi antartingkat wavelet. Pada data Susenas, `survey_rules.py` menerapkan aturan lompatan kuesioner agar sel *structural missing* tidak keliru diisi sebagai *item nonresponse*.

Evaluasi mencakup KDD Cup 2018, Guangzhou Traffic, PhysioNet 2012, dan Susenas. Metrik yang digunakan meliputi MAE, RMSE, CRPS, serta MAPE untuk evaluasi pengeluaran Susenas.

## Struktur repositori

```text
.streamlit/              Konfigurasi aplikasi Streamlit
Data/                    Direktori data untuk aplikasi atau eksperimen
hwd_app/                 Komponen aplikasi web HWD
layers/                  Lapisan pendukung model
  Diff_layers.py         Komponen proses difusi
  Embed.py               Komponen embedding
  __init__.py            Inisialisasi modul Python
models/                  Implementasi model
  diffusion.py           Proses difusi
  main_model.py          Model utama HWD
  ts_transformer.py      Transformer deret waktu
  __init__.py            Inisialisasi modul Python
HWD_Colab.ipynb          Notebook eksperimen benchmark
HWD_Colab_susenas.ipynb  Notebook eksperimen Susenas
dataset.py               Pemuatan dan penyiapan dataset
generate_missing.py      Pembentukan pola data hilang
incremental.py           Kode eksperimen incremental
survey_rules.py          Aturan kuesioner (*survey rule engine*)
train.py                 Pelatihan model
utils.py                 Fungsi pendukung
.gitignore               Aturan berkas yang diabaikan Git
```

## Dataset dan referensi

| Dataset | Sumber |
|---|---|
| KDD Cup 2018 | [KDD Cup](http://www.kdd.org/kdd2018/) |
| Guangzhou Traffic | [Zenodo: Guangzhou Traffic](https://zenodo.org/record/1205229) |
| PhysioNet 2012 | [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) |
| Susenas | Data survei Badan Pusat Statistik; penggunaan mengikuti ketentuan akses dan kerahasiaan data yang berlaku. |

## Konfigurasi utama HWD

| Hyperparameter | Nilai |
|---|---:|
| Panjang *window* (`L`) | 48 |
| *Mother wavelet* | Daubechies-2 |
| Tingkat dekomposisi DWT | 3 |
| Encoder aproksimasi | 8 lapisan |
| Encoder detail | 4 lapisan per tingkat |
| `d_model` / `channel` | 128 / 128 |
| Jumlah *attention heads* | 8 |
| *Feed-forward* encoder wavelet / *dropout* | 128 / 0,1 |
| `residual_layers` | 4 |
| *Feed-forward* temporal attention | 64 |
| `proj_t` (proyeksi Linformer) | 128 |
| *Time embedding* / *feature embedding* | 128 / 16 |
| Langkah difusi (`T`) | 50 |
| Jadwal *noise* | Kuadratik, β = 0,0001–0,2 |
| Optimizer / *learning rate* | Adam / 0,001 |
| *Weight decay* | 0,000001 |
| *Learning-rate scheduler* | MultiStepLR pada 75% dan 90% epoch, γ = 0,1 |

Parameter di atas merupakan ringkasan konfigurasi penelitian. Detail konfigurasi eksperimen dapat diperiksa pada notebook dan skrip terkait.

## Lingkungan komputasi dan waktu proses

Eksperimen benchmark dijalankan di Google Colab dengan GPU NVIDIA Tesla T4 bermemori 16 GB. Eksperimen Susenas dijalankan di Kaggle menggunakan 2 × NVIDIA Tesla T4. Lingkungan perangkat lunak utama menggunakan PyTorch 2.4, PyWavelets 1.5 untuk DWT, dan `properscoring` untuk perhitungan CRPS.

| Dataset | GPU | Waktu latih per epoch | Total pelatihan | Total inferensi | Sampel inferensi |
|---|---|---:|---:|---:|---:|
| KDD Cup 2018 | Tesla T4 | 4,7–7,1 detik | ±0,6 jam | 1,1–1,4 jam | 100 |
| Guangzhou Traffic | Tesla T4 | 11,7–14,3 detik | 2,7–4,0 jam | 2,7–3,3 jam | 100 |
| PhysioNet 2012 | Tesla T4 | 129–145 detik | ±1,9 jam | ±3,1 jam | 10 |
| Susenas | 2 × Tesla T4 | 145–186 detik | 4,0–5,2 jam | 7,9–8,1 jam | 10 |

Rentang waktu mencakup seluruh skenario pada setiap dataset. Inferensi menggunakan 50 langkah difusi per sampel, sehingga kebutuhan komputasi meningkat seiring jumlah *window* dan sampel. Untuk Susenas, inferensi mencakup 335.000 *forward pass* pada 21.432 *window*. Pengolahan data berukuran besar sebaiknya dijalankan pada lingkungan dengan GPU dan memori yang memadai. Aplikasi web publik lebih sesuai untuk demonstrasi dan data berukuran kecil.

## Menjalankan aplikasi dan eksperimen

- **Mencoba aplikasi:** buka [hwd-new.streamlit.app](https://hwd-new.streamlit.app/).
- **Menjalankan eksperimen:** gunakan notebook `HWD_Colab.ipynb` untuk eksperimen benchmark dan `HWD_Colab_susenas.ipynb` untuk eksperimen Susenas.
- **Menjalankan pelatihan melalui skrip:** tinjau `train.py`, `dataset.py`, `generate_missing.py`, serta modul pada `models/` dan `layers/`.

Pastikan dependensi, lokasi data, dan parameter telah disesuaikan dengan lingkungan yang digunakan. Struktur kolom dan praproses harus mengikuti pipeline eksperimen agar hasilnya dapat dibandingkan secara tepat.

## Catatan penggunaan

- Gunakan data Susenas sesuai ketentuan akses, kerahasiaan, dan pemanfaatan data yang berlaku.
- Hasil imputasi merupakan nilai taksiran, bukan jawaban observasi asli. Evaluasi kewajaran hasil sebelum digunakan dalam analisis atau tabulasi.
- Aturan pada `survey_rules.py` perlu disesuaikan jika diterapkan pada versi kuesioner atau struktur data yang berbeda.
- Untuk analisis yang memerlukan pengukuran ketidakpastian, pertimbangkan penggunaan *multiple imputation* sesuai tujuan analisis.
