"""
HWD Imputation Explorer
Aplikasi Streamlit untuk demonstrasi model Hierarchical Wavelet Diffusion (HWD)
Cloudya Qashwah Montolalu, Politeknik Statistik STIS, 2026

Revisi:
  - Konfigurasi mengikuti BAB III final: Daubechies-2, tiga tingkat (8, 8, 14, 25).
  - Tidak ada angka atau imputasi rekaan. Angka pembanding diambil dari
    Yang et al. (2024) Tabel 1; angka HWD dibaca dari results/hwd_results.csv.
  - Halaman visualisasi membaca berkas keluaran notebook (npy/csv) yang nyata.
  - Rule engine memakai aturan survey_rules.py (R703_A, R705, R706, R707).
  - Demo inferensi: HWD dengan checkpoint (T = 50 tetap), atau pembanding
    sederhana yang diberi label jelas. Mode simulasi yang memakai nilai
    sebenarnya sudah dihapus.
"""

import io
import json
import math
import time
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from hwd_core import (
    APP_DIR, REPO_DIR, SEQ_LEN, SENTINEL, RATES,
    C_OBS, C_EVAL, C_NAT, C_SPAN, C_HWD, C_FGTI, C_CSDI, C_MEAN,
    HWD_SPEC, DATASET_INFO, BASELINES, DEFAULT_HWD_RESULTS, ABLATION_LABEL,
    RULE_SOURCE, prepare_susenas, valid_mask_fn, restore_fn, _is_filled,
    idn, load_npy, default_col_names, make_windows, make_mask, mae_rmse,
    plot_series, plot_scatter, load_hwd_checkpoint, run_hwd,
    baseline_mean, baseline_interp,
    susenas_prepare, susenas_mask, susenas_compose, susenas_metrics, plot_susenas_scatter,
)


@st.cache_data(show_spinner=False)
def load_hwd_results(uploaded_bytes=None):
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes)), "berkas yang diunggah"
    path = APP_DIR / "results" / "hwd_results.csv"
    if path.exists():
        return pd.read_csv(path), "results/hwd_results.csv"
    return DEFAULT_HWD_RESULTS.copy(), "nilai bawaan app.py"



# ═════════════════════════════════════════════════════════════════════════════
#  TAMPILAN
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="HWD Imputation Explorer", page_icon="〰️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header {background: linear-gradient(135deg, #085041 0%, #0F6E56 55%, #1D9E75 100%);
  padding: 2.2rem 2rem 1.8rem; border-radius: 12px; margin-bottom: 1.6rem; color: white;}
.main-header h1 {font-size: 1.9rem; font-weight: 700; margin: 0 0 .35rem 0;}
.main-header p {font-size: .95rem; opacity: .9; margin: 0;}
.badge {display:inline-block; background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3);
  border-radius: 20px; padding: 2px 10px; font-size: .75rem; margin: 10px 6px 0 0;}
.card {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:1rem 1.2rem;}
.card .label {font-size:.72rem; color:#64748B; text-transform:uppercase; letter-spacing:.05em;}
.card .value {font-size:1.5rem; font-weight:700; color:#0F6E56;}
.section-title {font-size:1.02rem; font-weight:600; color:#1E293B; margin:.4rem 0 .8rem;
  padding-bottom:.4rem; border-bottom:2px solid #E2E8F0;}
.note {background:#F0FDF9; border-left:4px solid #0F6E56; padding:.8rem 1rem;
  border-radius:0 8px 8px 0; font-size:.88rem; color:#064E3B; margin:.8rem 0;}
.warn {background:#FEF6E7; border-left:4px solid #BA7517; padding:.8rem 1rem;
  border-radius:0 8px 8px 0; font-size:.88rem; color:#633806; margin:.8rem 0;}
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 〰️ HWD Explorer")
    page = st.radio("Navigasi", ["Tentang Model", "Benchmark", "Visualisasi Hasil",
                                 "Rule Engine Susenas", "Imputasi Susenas", "Demo Inferensi"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption("Hierarchical Wavelet Diffusion · Cloudya Qashwah Montolalu · "
               "Politeknik Statistik STIS · 2026")
    st.caption(f"Kode model: {'ditemukan di ' + REPO_DIR.name if REPO_DIR else 'tidak ditemukan'}")


def card(label, value):
    st.markdown(f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HALAMAN 1 — TENTANG MODEL
# ─────────────────────────────────────────────────────────────────────────────
if page == "Tentang Model":
    st.markdown("""
    <div class="main-header">
      <h1>HWD Imputation Explorer</h1>
      <p>Hierarchical Wavelet Diffusion untuk imputasi probabilistik data deret waktu multivariat</p>
      <span class="badge">Conditional diffusion</span><span class="badge">Daubechies-2, 3 tingkat</span>
      <span class="badge">Cross-level attention</span><span class="badge">Survey rule engine</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-title">Cara kerja HWD</div>', unsafe_allow_html=True)
        st.markdown("""
HWD dibangun di atas basis kode FGTI (Yang et al., 2024) dengan tiga perubahan.

**Wavelet conditioning.** Nilai kondisi diurai dengan DWT Daubechies-2 tiga tingkat menjadi
koefisien aproksimasi A₃ (tren global) serta detail D₃, D₂, dan D₁ (variasi kasar sampai halus).
Pada window 48 titik, panjang keempat cabang adalah 8, 8, 14, dan 25 titik. Daubechies-4 tidak
dipakai karena filternya yang lebih panjang hanya menyisakan satu tingkat dekomposisi pada L = 48.

**Cross-level attention.** Setiap encoder detail menerima konteks dari tingkat di atasnya
secara berurutan, dari A₃ ke D₁, sehingga rekonstruksi fluktuasi halus mengikuti tren global.

**Survey rule engine.** Khusus Susenas, posisi structural missing akibat aturan lompatan
kuesioner dikosongkan kembali setelah imputasi, sehingga hanya item nonresponse yang terisi.

Komponen lain (jaringan denoising, jadwal noise, prosedur pelatihan) mengikuti FGTI, sehingga
selisih kinerja dapat ditelusuri ke jalur conditioning.
        """)
        img = APP_DIR / "assets" / "gambar_3_3_arsitektur.png"
        if img.exists():
            st.image(str(img), caption="Arsitektur HWD (Gambar 3.3)", width="stretch")
        else:
            st.caption("Letakkan ekspor Gambar 3.3 di assets/gambar_3_3_arsitektur.png "
                       "untuk menampilkan diagram arsitektur di sini.")
    with c2:
        st.markdown('<div class="section-title">Spesifikasi (Tabel 3.5)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(HWD_SPEC, columns=["Komponen", "Nilai"]),
                     hide_index=True, width="stretch")

    st.markdown('<div class="section-title">Dataset dan konfigurasi per dataset (Tabel 3.6)</div>',
                unsafe_allow_html=True)
    st.dataframe(DATASET_INFO, hide_index=True, width="stretch")
    st.markdown('<div class="note">Seluruh metrik dihitung pada skala Z-score dan hanya pada posisi '
                'artificial missing (Ω<sub>e</sub>). Posisi natural missing dan structural missing '
                'tidak memiliki nilai sebenarnya sehingga tidak dinilai.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HALAMAN 2 — BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Benchmark":
    st.markdown("## Hasil benchmark")
    with st.expander("Perbarui hasil HWD dari berkas CSV"):
        st.caption("Kolom: dataset, mekanisme, missing_rate, mode, konfigurasi, MAE, RMSE, CRPS, catatan")
        up = st.file_uploader("hwd_results.csv", type=["csv"], key="res_csv")
    res, res_src = load_hwd_results(up.getvalue() if up else None)
    res["missing_rate"] = pd.to_numeric(res["missing_rate"], errors="coerce")

    c1, c2 = st.columns(2)
    dataset = c1.selectbox("Dataset", list(BASELINES.keys()))
    metric = c2.selectbox("Metrik", ["MAE", "RMSE", "CRPS"])

    hwd_rows = res[(res["dataset"] == dataset) & (res["mekanisme"].str.upper() == "MCAR")
                   & (res["mode"] == "benchmark") & (res["konfigurasi"] == "full")]
    hwd_vals = [hwd_rows.loc[hwd_rows["missing_rate"] == r, metric].mean()
                if (hwd_rows["missing_rate"] == r).any() else np.nan for r in RATES]

    table = {}
    if metric in BASELINES[dataset]:
        for mth, vals in BASELINES[dataset][metric].items():
            table[mth] = vals
    table["HWD"] = hwd_vals
    df_tab = pd.DataFrame(table, index=[f"{r}%" for r in RATES]).T

    cc, ct = st.columns([3, 2])
    with cc:
        series = {m: np.array(v, dtype=float) for m, v in table.items()
                  if not np.all(np.isnan(np.array(v, dtype=float)))}
        if not series:
            st.info(f"Belum ada hasil {metric} untuk {dataset}.")
        else:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            colors = {"Mean": C_MEAN, "CSDI": C_CSDI, "FGTI": C_FGTI, "HWD": C_HWD}
            for mth, y in series.items():
                ax.plot(RATES, y, marker="o", lw=2.4 if mth == "HWD" else 1.4,
                        ls="-" if mth == "HWD" else "--", color=colors[mth], label=mth)
            ax.set_xticks(RATES)
            ax.set_xlabel("Tingkat missing (%)")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.3, ls="--")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(fontsize=8, frameon=False)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
    with ct:
        st.dataframe(df_tab.map(lambda v: idn(v, 4)), width="stretch")
        fg = BASELINES[dataset][metric]["FGTI"]
        lines = []
        for r, h, f in zip(RATES, hwd_vals, fg):
            if not np.isnan(h):
                d = (f - h) / f * 100
                lines.append(f"{r}%: HWD {'lebih baik' if d > 0 else 'lebih buruk'} "
                             f"{idn(abs(d), 1)}% dari FGTI")
        if lines:
            st.markdown('<div class="note">' + "<br>".join(lines) + "</div>",
                        unsafe_allow_html=True)
        if metric == "CRPS":
            st.caption("Mean imputation tidak menghasilkan sebaran, sehingga CRPS-nya tidak ada.")
        n_na = int(np.isnan(np.array(hwd_vals, dtype=float)).sum())
        if n_na:
            st.markdown(f'<div class="warn">{n_na} dari 4 tingkat missing HWD belum tersedia. '
                        'Tambahkan hasilnya ke results/hwd_results.csv.</div>',
                        unsafe_allow_html=True)
    st.caption(f"Sumber: Mean, CSDI, FGTI dari Yang et al. (2024), MAE dan RMSE Tabel 1, CRPS Tabel 5 (MCAR). "
               f"HWD dari eksperimen penelitian ini ({res_src}), mode benchmark.")

    st.markdown('<div class="section-title">Ablation study (Tabel 4.4)</div>', unsafe_allow_html=True)
    abl = res[res["konfigurasi"].isin(ABLATION_LABEL.keys())]
    abl = abl[(abl["dataset"] == "KDD Cup 2018") & (abl["missing_rate"] == 10)]
    if len(abl):
        full = abl[abl["konfigurasi"] == "full"]
        rows = []
        for k, lab in ABLATION_LABEL.items():
            r = abl[abl["konfigurasi"] == k]
            if not len(r):
                continue
            row = {"Konfigurasi": lab}
            for m in ["MAE", "RMSE", "CRPS"]:
                v = float(r[m].iloc[0])
                row[m] = idn(v, 4)
                if k != "full" and len(full):
                    f = float(full[m].iloc[0])
                    row[f"Δ{m} vs lengkap"] = f"+{idn((v - f) / f * 100, 1)}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).fillna(""), hide_index=True, width="stretch")
        st.caption("KDD Cup 2018, MCAR 10%, Daubechies-2 tiga tingkat. Nilai positif berarti "
                   "galat naik ketika komponen dihapus.")

    sus = res[res["dataset"].str.contains("Susenas", case=False, na=False)]
    if len(sus):
        st.markdown('<div class="section-title">Susenas KP</div>', unsafe_allow_html=True)
        st.dataframe(sus, hide_index=True, width="stretch")
        st.caption("CRPS Susenas dihitung dengan properscoring dan tidak sebanding dengan CRPS benchmark.")

# ─────────────────────────────────────────────────────────────────────────────
#  HALAMAN 3 — VISUALISASI HASIL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Visualisasi Hasil":
    st.markdown("## Visualisasi hasil imputasi")
    st.markdown('<div class="note">Halaman ini membaca berkas keluaran notebook eksperimen, '
                'sehingga yang ditampilkan adalah hasil imputasi yang sebenarnya.</div>',
                unsafe_allow_html=True)

    src = st.radio("Sumber", ["Array NumPy (.npy)", "CSV evaluasi"], horizontal=True)

    if src == "Array NumPy (.npy)":
        c1, c2 = st.columns(2)
        f_data = c1.file_uploader("data_{stem}_norm.npy", type=["npy"])
        f_imp = c2.file_uploader("imputed_{stem}_norm.npy", type=["npy"])
        c3, c4 = st.columns(2)
        f_ev = c3.file_uploader("evalmask_{stem}.npy", type=["npy"])
        f_cm = c4.file_uploader("condmask_{stem}.npy (opsional)", type=["npy"])
        with st.expander("Opsional: denormalisasi ke skala asli"):
            f_mu = st.file_uploader("means.npy", type=["npy"])
            f_sd = st.file_uploader("stds.npy", type=["npy"])

        if not (f_data and f_imp and f_ev):
            st.info("Unggah minimal tiga berkas: data, imputed, dan evalmask. Berkas ini disimpan "
                    "oleh sel ‘Simpan + line chart’ di notebook KDD maupun Susenas.")
        else:
            data, imp, ev = load_npy(f_data), load_npy(f_imp), load_npy(f_ev).astype(bool)
            cm = load_npy(f_cm) if f_cm else None
            if data.shape != imp.shape or data.shape != ev.shape:
                st.error(f"Bentuk tidak cocok: data {data.shape}, imputed {imp.shape}, evalmask {ev.shape}.")
                st.stop()
            N, L, K = data.shape
            names = default_col_names(K)
            ylabel = "Z-score"
            if f_mu and f_sd:
                mu, sd = load_npy(f_mu).reshape(-1), load_npy(f_sd).reshape(-1)
                if len(mu) == K and len(sd) == K:
                    data = np.where(data <= SENTINEL + 10, SENTINEL, data * sd + mu)
                    imp = imp * sd + mu
                    ylabel = "Skala asli"
                else:
                    st.warning("Panjang means/stds tidak sama dengan K; denormalisasi dilewati.")

            mae, rmse = mae_rmse(data, imp, ev)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                card("Bentuk", f"{N} × {L} × {K}")
            with m2:
                card("Posisi uji Ω<sub>e</sub>", f"{int(ev.sum()):,}".replace(",", "."))
            with m3:
                card(f"MAE ({ylabel})", idn(mae, 4))
            with m4:
                card(f"RMSE ({ylabel})", idn(rmse, 4))

            s1, s2 = st.columns([2, 1])
            pick = s1.multiselect("Variabel", list(range(K)), default=list(range(min(3, K))),
                                  format_func=lambda k: names[k], max_selections=6)
            score = ev[:, :, pick].sum(axis=(1, 2)) - (data[:, :, pick] <= SENTINEL + 10).sum(axis=(1, 2)) \
                if pick else np.zeros(N)
            w = s2.number_input("Window", 0, N - 1, int(np.argmax(score)) if pick else 0,
                                help="Nilai awal: window dengan posisi uji terbanyak.")
            if pick:
                fig = plot_series(data, imp, ev, cm, int(w), pick, names, ylabel)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                st.markdown('<div class="section-title">Nilai asli vs imputasi di Ω<sub>e</sub></div>',
                            unsafe_allow_html=True)
                cols = st.columns(min(3, len(pick)))
                for i, k in enumerate(pick[:3]):
                    msk = ev[:, :, k]
                    if msk.sum() > 2:
                        f2 = plot_scatter(data[:, :, k][msk], imp[:, :, k][msk], names[k])
                        cols[i].pyplot(f2, width="stretch")
                        plt.close(f2)

            per = []
            for k in range(K):
                a, b = mae_rmse(data[:, :, k], imp[:, :, k], ev[:, :, k])
                per.append([names[k], int(ev[:, :, k].sum()), a, b])
            dper = pd.DataFrame(per, columns=["Variabel", "n Ω_e", "MAE", "RMSE"]).dropna()
            st.markdown('<div class="section-title">Galat per variabel</div>', unsafe_allow_html=True)
            st.dataframe(dper.sort_values("MAE"), hide_index=True, width="stretch")

    else:
        f_csv = st.file_uploader("evaluasi_{stem}_norm.csv atau _asli.csv", type=["csv"])
        if not f_csv:
            st.info("Unggah CSV evaluasi dari notebook (kolom: variabel, nilai_asli, imputasi_hwd).")
        else:
            ev_df = pd.read_csv(f_csv)
            need = {"variabel", "nilai_asli", "imputasi_hwd"}
            if not need.issubset(ev_df.columns):
                st.error(f"Kolom wajib tidak lengkap: {sorted(need - set(ev_df.columns))}")
                st.stop()
            ev_df = ev_df.dropna(subset=["nilai_asli", "imputasi_hwd"])
            ev_df["abs"] = (ev_df["imputasi_hwd"] - ev_df["nilai_asli"]).abs()
            ev_df["sq"] = (ev_df["imputasi_hwd"] - ev_df["nilai_asli"]) ** 2
            summ = (ev_df.groupby("variabel").agg(n=("abs", "size"), MAE=("abs", "mean"),
                                                  RMSE=("sq", lambda s: np.sqrt(s.mean())))
                    .sort_values("MAE").reset_index())
            st.dataframe(summ, hide_index=True, width="stretch")
            if "tahun" in ev_df.columns:
                st.caption("Kolom tahun tersedia: saring per tahun di bawah.")
                th = st.selectbox("Tahun", ["Semua"] + sorted(ev_df["tahun"].dropna().unique().tolist()))
                if th != "Semua":
                    ev_df = ev_df[ev_df["tahun"] == th]
            v = st.selectbox("Variabel untuk scatter", summ["variabel"].tolist())
            sub = ev_df[ev_df["variabel"] == v]
            if len(sub) > 2:
                f2 = plot_scatter(sub["nilai_asli"].to_numpy(), sub["imputasi_hwd"].to_numpy(), v)
                st.pyplot(f2, width="content")
                plt.close(f2)

# ─────────────────────────────────────────────────────────────────────────────
#  HALAMAN 4 — RULE ENGINE SUSENAS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Rule Engine Susenas":
    st.markdown("## Survey rule engine")
    st.markdown('<div class="note">Sebagian pertanyaan Susenas hanya ditanyakan kepada responden '
                'dengan kondisi tertentu. Sel yang kosong karena aturan ini adalah <b>structural '
                'missing</b> dan harus tetap kosong, berbeda dengan <b>item nonresponse</b> yang '
                'diimputasi.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Aturan yang diterapkan</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ["R703_A terisi ‘A’ (KRT bekerja seminggu terakhir)", "R705"],
        ["KRT tidak bekerja: R703_A kosong dan R705 ≠ 1", "R706, R707"],
    ], columns=["Kondisi pemicu", "Kolom yang menjadi structural missing"]),
        hide_index=True, width="stretch")
    st.caption(f"Sumber aturan: {RULE_SOURCE}. Status bekerja = R703_A terisi ‘A’ atau R705 = 1. "
               "Aturan dievaluasi pada data asli, bukan pada data hasil imputasi.")

    st.markdown('<div class="section-title">Uji aturan secara interaktif</div>', unsafe_allow_html=True)
    demo = pd.DataFrame({
        "R703_A": ["A", "", "", "A", ""],
        "R705": [np.nan, 1, 5, np.nan, 5],
        "R706": [47, 10, np.nan, np.nan, 3],
        "R707": [3, 1, np.nan, 4, np.nan],
        "food": [2450000, 1890000, np.nan, 3120000, 980000],
    }, index=["RT A", "RT B", "RT C", "RT D", "RT E"])
    edited = st.data_editor(demo, width="stretch", num_rows="dynamic", key="rule_editor")

    prep = prepare_susenas(edited.reset_index(drop=True))
    valid = valid_mask_fn(prep)
    view_cols = [c for c in ["R703_A", "R705", "R706", "R707", "food"] if c in prep.columns]
    col_pos = {c: list(prep.columns).index(c) for c in view_cols}

    def _status(i, c):
        val = prep[c].iloc[i]
        empty = (pd.isna(val) or (isinstance(val, str) and val.strip() == ""))
        if c == "R703_A":
            return "pemicu: tidak dilingkari" if empty else "pemicu: bekerja"
        if not valid[i, col_pos[c]]:
            return "structural" if empty else "structural terisi (tidak konsisten)"
        return "item nonresponse → diimputasi" if empty else "terisi"

    stat = pd.DataFrame({c: [_status(i, c) for i in range(len(prep))] for c in view_cols},
                        index=edited.index[:len(prep)])
    palette = {"structural": "background-color:#E1F5EE;color:#085041",
               "structural terisi (tidak konsisten)": "background-color:#FCEBEB;color:#791F1F",
               "item nonresponse → diimputasi": "background-color:#FEF6E7;color:#633806",
               "terisi": "",
               "pemicu: bekerja": "color:#444441;font-style:italic",
               "pemicu: tidak dilingkari": "color:#444441;font-style:italic"}
    st.dataframe(stat.style.map(lambda v: palette.get(v, "")), width="stretch")
    st.caption("Teal: harus kosong. Kuning: kosong yang akan diimputasi. Merah: sel terisi padahal "
               "seharusnya kosong menurut aturan.")

    st.markdown('<div class="section-title">Statistik dari data asli</div>', unsafe_allow_html=True)
    f_sus = st.file_uploader("CSV Susenas asli (dengan kolom R703_A, R705, R706, R707)", type=["csv"])
    if f_sus:
        raw = pd.read_csv(f_sus, low_memory=False)
        p2 = prepare_susenas(raw)
        v2 = valid_mask_fn(p2)
        rows = []
        for c in ["R705", "R706", "R707"]:
            if c not in p2.columns:
                continue
            j = list(p2.columns).index(c)
            empty = ~_is_filled(p2[c]).to_numpy()
            struct = ~v2[:, j]
            rows.append([c, int(struct.sum()), int((struct & ~empty).sum()),
                         int((empty & ~struct).sum())])
        st.dataframe(pd.DataFrame(rows, columns=["Kolom", "Structural missing",
                                                 "Structural tetapi terisi",
                                                 "Item nonresponse"]),
                     hide_index=True, width="stretch")
        st.caption(f"{len(raw):,} baris dianalisis.".replace(",", "."))

# ─────────────────────────────────────────────────────────────────────────────
#  HALAMAN — IMPUTASI SUSENAS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Imputasi Susenas":
    st.markdown("## Imputasi Susenas dengan HWD")
    st.markdown('<div class="note">Unggah data Susenas, lalu pilih salah satu mode. <b>Isi missing alami</b> '
                'mengisi sel yang kosong dan menghasilkan berkas lengkap. <b>Uji dengan missing buatan</b> '
                'menyembunyikan sebagian nilai yang diketahui, mengisinya kembali, lalu menampilkan metrik dan '
                'scatter perbandingan. Data hanya diproses selama sesi ini dan tidak disimpan aplikasi.</div>',
                unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Memuat model HWD…")
    def _load_sus_model(key, _ckpt_bytes, K):
        return load_hwd_checkpoint(_ckpt_bytes, K)

    # ── 1. Model ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">1. Model</div>', unsafe_allow_html=True)
    mdir = APP_DIR / "model_susenas"
    b_pt = sorted(mdir.glob("*.pt")) if mdir.is_dir() else []
    b_js = sorted(mdir.glob("*.json")) if mdir.is_dir() else []
    ckpt_bytes, paket = None, None
    if b_pt and b_js:
        ckpt_bytes = b_pt[0].read_bytes()
        paket = json.loads(b_js[0].read_text(encoding="utf-8"))
        st.caption(f"Memakai model bawaan aplikasi: {b_pt[0].name} dan {b_js[0].name}")
    else:
        m1, m2 = st.columns(2)
        f_pt = m1.file_uploader("Checkpoint HWD Susenas (.pt)", type=["pt"], key="sus_pt")
        f_js = m2.file_uploader("Paket model (.json)", type=["json"], key="sus_js")
        if f_pt:
            ckpt_bytes = f_pt.getvalue()
        if f_js:
            paket = json.loads(f_js.getvalue().decode("utf-8"))
    if paket:
        st.caption(f"Tahun latih {paket.get('train_years', '–')} · mode {paket.get('mode', '–')} · "
                   f"{len(paket['feat_cols'])} variabel · batas outlier 3×IQR untuk "
                   f"{len(paket.get('fences', {}))} variabel")

    # ── 2. Data ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">2. Data</div>', unsafe_allow_html=True)
    f_data = st.file_uploader("Data Susenas (.csv atau .xlsx)", type=["csv", "xlsx"], key="sus_data")
    if not (ckpt_bytes and paket and f_data):
        st.info("Siapkan checkpoint, paket model, dan data. Data harus memuat kolom "
                + ", ".join(paket["feat_cols"] if paket else ["R101", "…", "KAPITA"])
                + ", serta R703_A agar survey rule engine dapat dijalankan.")
        st.stop()
    df_in = (pd.read_excel(io.BytesIO(f_data.getvalue())) if f_data.name.lower().endswith(".xlsx")
             else pd.read_csv(io.BytesIO(f_data.getvalue()), low_memory=False))
    try:
        prep = susenas_prepare(df_in, paket)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    W = prep["data"].shape[0]
    st.caption(f"{prep['n']} baris · {W} window × {prep['L']} rumah tangga"
               + (f" (termasuk {prep['pad']} baris padding yang dibuang setelah imputasi)" if prep["pad"] else "")
               + f" · {int(prep['nat'].sum())} sel kosong awal · {int(prep['out'].sum())} sel outlier")
    with st.expander("Pratinjau data"):
        st.dataframe(df_in.head(20), width="stretch")

    # ── 3. Pengaturan ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">3. Pengaturan</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Isi missing alami", "Uji dengan missing buatan"], horizontal=True)
    mech, rate, seed = None, None, 1
    if mode == "Uji dengan missing buatan":
        p1, p2, p3 = st.columns(3)
        mech = p1.selectbox("Mekanisme", ["MCAR", "MAR", "block"])
        rate = p2.select_slider("Tingkat missing", [0.1, 0.2, 0.3, 0.4], value=0.1,
                                format_func=lambda v: f"{int(v * 100)}%")
        seed = p3.number_input("Seed", 0, 99999, 1)
    o1, o2, o3 = st.columns(3)
    n_samples = o1.slider("Jumlah sampel", 1, 20, 10, help="Eksperimen skripsi memakai 10 sampel.")
    replace_out = o2.checkbox("Ganti nilai outlier dengan hasil imputasi", value=False,
                              help="Bawaannya mati: nilai outlier asli dipertahankan di berkas keluaran.")
    snap = o3.checkbox("Bulatkan variabel kode ke kode yang sah", value=True)
    st.caption(f"Beban komputasi: {math.ceil(W / 8) * n_samples * 50:,} forward pass. ".replace(",", ".")
               + "Di CPU, perkiraannya sekitar setengah sampai dua menit untuk 10 sampel.")

    sig = (f_data.name, f_data.size, mode, mech, rate, int(seed), n_samples, replace_out, snap,
           hashlib.md5(ckpt_bytes).hexdigest())
    if st.button("Jalankan imputasi", type="primary", width="stretch"):
        cond = prep["gt"].copy() if mode == "Isi missing alami" else susenas_mask(prep, mech, rate, seed)
        ev = (prep["gt"] == 1) & (cond == 0)
        try:
            model, meta, _ = _load_sus_model(sig[-1], ckpt_bytes, prep["data"].shape[2])
        except ModuleNotFoundError as e:
            st.error(f"Kode model atau PyTorch tidak ditemukan ({e}). Pastikan folder models/ dan layers/ "
                     "ada di repo yang sama dengan aplikasi.")
            st.stop()
        except Exception as e:
            st.error(f"Checkpoint tidak dapat dimuat: {e}")
            st.stop()
        t0 = time.time()
        bar = st.progress(0.0, text="Menjalankan inferensi…")
        pred, crps = run_hwd(model, prep["data"], cond, prep["gt"], n_samples, 8,
                             lambda q, t: bar.progress(q, text=t))
        out_df, info = susenas_compose(df_in, prep, pred, cond, replace_outlier=replace_out, snap_codes=snap)
        result = {"mode": mode, "out": out_df, "info": info, "detik": time.time() - t0, "crps": crps}
        if mode != "Isi missing alami":
            result["metrik"], result["tabel"], result["pairs"], result["evaluasi"] = \
                susenas_metrics(prep, pred, ev)
        st.session_state["sus_result"], st.session_state["sus_sig"] = result, sig

    # ── 4. Hasil ────────────────────────────────────────────────────────────
    r = st.session_state.get("sus_result") if st.session_state.get("sus_sig") == sig else None
    if r:
        st.markdown('<div class="section-title">4. Hasil</div>', unsafe_allow_html=True)
        info = r["info"]
        if r["mode"] == "Isi missing alami":
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Sel diimputasi", f"{info['diimputasi']:,}".replace(",", "."))
            with c2:
                card("Outlier ditandai", f"{info['outlier']:,}".replace(",", "."))
            with c3:
                card("Structural dikosongkan", f"{info['structural']:,}".replace(",", "."))
            st.caption(f"Selesai dalam {r['detik']:.1f} detik. Kolom _diimputasi dan _outlier pada berkas "
                       "keluaran menandai variabel yang diisi dan yang terdeteksi sebagai outlier di setiap baris.")
        else:
            m = r["metrik"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card("Posisi uji", f"{m['n_uji']:,}".replace(",", "."))
            with c2:
                card("MAE (Z-score)", idn(m["MAE_z"], 4))
            with c3:
                card("RMSE (Z-score)", idn(m["RMSE_z"], 4))
            with c4:
                card("CRPS (Z-score)", idn(r["crps"], 4))
            tab = r["tabel"].copy()
            if len(tab):
                small = int(tab["n"].min()) < 30
                for col_ in ["MAE (Rp)", "RMSE (Rp)"]:
                    tab[col_] = tab[col_].map(lambda v: f"{v:,.0f}".replace(",", "."))
                tab["MAPE (%)"] = tab["MAPE (%)"].map(lambda v: idn(v, 2))
                st.markdown('<div class="section-title">Galat per variabel pengeluaran</div>',
                            unsafe_allow_html=True)
                st.dataframe(tab, hide_index=True, width="stretch")
                if small:
                    st.markdown('<div class="warn">Jumlah posisi uji per variabel masih sedikit, sehingga metrik '
                                'bersifat ilustratif. Naikkan tingkat missing atau unggah lebih banyak baris untuk '
                                'hasil yang lebih stabil.</div>', unsafe_allow_html=True)
                fig = plot_susenas_scatter(r["pairs"])
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            st.caption(f"Selesai dalam {r['detik']:.1f} detik. Metrik skala Z dihitung atas seluruh posisi uji; "
                       "tabel dan scatter khusus variabel pengeluaran dalam rupiah.")
            st.download_button("Unduh posisi uji dan hasil imputasinya (.csv)",
                               r["evaluasi"].to_csv(index=False).encode("utf-8"),
                               file_name="evaluasi_imputasi_susenas.csv", mime="text/csv", width="stretch")
        st.download_button("Unduh data hasil imputasi (.csv)", r["out"].to_csv(index=False).encode("utf-8"),
                           file_name="hasil_imputasi_susenas.csv", mime="text/csv", width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
#  HALAMAN 5 — DEMO INFERENSI
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Demo Inferensi":
    st.markdown("## Demo inferensi")
    st.markdown('<div class="note">Unggah data, sembunyikan sebagian nilai secara artifisial, lalu '
                'isi dengan HWD (butuh checkpoint) atau dengan metode pembanding sederhana. '
                'Metrik dihitung pada posisi yang disembunyikan.</div>', unsafe_allow_html=True)

    f_csv = st.file_uploader("CSV data (baris = waktu atau rumah tangga, kolom = variabel)", type=["csv"])
    if not f_csv:
        st.info("Unggah CSV untuk memulai. Minimal 48 baris; sel kosong dianggap missing alami.")
        st.stop()

    df_up = pd.read_csv(f_csv, low_memory=False)
    num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
    use_cols = st.multiselect("Kolom variabel (urutan harus sama dengan saat pelatihan)",
                              num_cols, default=num_cols)
    if not use_cols:
        st.stop()
    X = df_up[use_cols].to_numpy(dtype=float)
    if len(X) < SEQ_LEN:
        st.error("Data kurang dari 48 baris.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    method = c1.radio("Metode", ["HWD (checkpoint)", "Mean imputation", "Interpolasi linear"])
    mech = c2.selectbox("Mekanisme", ["MCAR", "MAR", "block"])
    rate = c2.select_slider("Tingkat missing", [0.1, 0.2, 0.3, 0.4], value=0.1,
                            format_func=lambda v: f"{int(v * 100)}%")
    excl = c3.multiselect("Kolom dikecualikan dari mask (mis. identitas)", use_cols,
                          default=[c for c in ["R101", "R102"] if c in use_cols])
    mar_col = c3.selectbox("Kolom penentu MAR", use_cols) if mech == "MAR" else None
    seed = c3.number_input("Seed", 0, 99999, 1)

    ckpt, f_mu, f_sd = None, None, None
    if method == "HWD (checkpoint)":
        h1, h2, h3 = st.columns(3)
        ckpt = h1.file_uploader("Checkpoint .pt", type=["pt"])
        f_mu = h2.file_uploader("means.npy (disarankan)", type=["npy"])
        f_sd = h3.file_uploader("stds.npy (disarankan)", type=["npy"])

    # Normalisasi
    K = X.shape[1]
    if f_mu and f_sd:
        mu, sd = load_npy(f_mu).reshape(-1), load_npy(f_sd).reshape(-1)
        if len(mu) != K:
            st.error(f"means.npy berisi {len(mu)} nilai, sedangkan kolom terpilih {K}.")
            st.stop()
        norm_src = "means/stds pelatihan"
    else:
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        norm_src = "statistik data unggahan"
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    mu = np.where(np.isnan(mu), 0.0, mu)
    Xn = (X - mu) / sd
    Xn[np.isnan(Xn)] = SENTINEL
    data_all, n_ok = make_windows(Xn)
    N_all = data_all.shape[0]

    n_win = st.slider("Jumlah window diproses", 1, N_all, min(N_all, 4),
                      help="Inferensi HWD di CPU lambat; mulai dari sedikit window.")
    data = data_all[:n_win]
    gt = (data != SENTINEL).astype(np.float32)
    ex_idx = [use_cols.index(c) for c in excl]
    cond = make_mask(data, gt, mech, rate, int(seed), ex_idx,
                     use_cols.index(mar_col) if mar_col else None)
    ev = (gt == 1) & (cond == 0)

    n_samples = 5
    if method == "HWD (checkpoint)":
        n_samples = st.slider("Jumlah sampel", 1, 20, 5)
        n_fp = int(np.ceil(n_win / 8)) * n_samples * 50
        st.caption(f"Perkiraan beban: {n_fp:,} forward pass (batch 8 × {n_samples} sampel × 50 langkah). "
                   .replace(",", ".") + f"Normalisasi memakai {norm_src}.")

    if st.button("Jalankan imputasi", type="primary", width="stretch"):
        crps = np.nan
        t0 = time.time()
        if method == "HWD (checkpoint)":
            if ckpt is None:
                st.error("Unggah checkpoint .pt terlebih dahulu.")
                st.stop()
            try:
                model, meta, unexpected = load_hwd_checkpoint(ckpt.getvalue(), K)
            except ModuleNotFoundError as e:
                st.error(f"Kode model atau PyTorch tidak ditemukan ({e}). Letakkan app di dalam repo "
                         "HWD (sejajar dengan folder models/ dan layers/) atau set variabel HWD_REPO.")
                st.stop()
            except Exception as e:
                st.error(f"Checkpoint tidak dapat dimuat: {e}")
                st.stop()
            if unexpected:
                st.caption(f"{len(unexpected)} bobot tambahan di checkpoint diabaikan "
                           f"(contoh: {unexpected[0]}).")
            bar = st.progress(0.0, text="Menjalankan inferensi…")
            pred, crps = run_hwd(model, data, cond, gt, n_samples, 8,
                                 lambda p, t: bar.progress(p, text=t))
            label = "HWD"
        elif method == "Mean imputation":
            pred, label = baseline_mean(data, cond), "Mean imputation"
        else:
            pred, label = baseline_interp(data, cond), "Interpolasi linear"
        elapsed = time.time() - t0

        mae, rmse = mae_rmse(data, pred, ev)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            card("Metode", label)
        with m2:
            card("MAE (Z-score)", idn(mae, 4))
        with m3:
            card("RMSE (Z-score)", idn(rmse, 4))
        with m4:
            card("CRPS (properscoring)", idn(crps, 4))
        st.caption(f"{int(ev.sum()):,} posisi uji · {n_win} window · {elapsed:.1f} detik".replace(",", "."))
        if label != "HWD":
            st.markdown('<div class="warn">Hasil ini berasal dari metode pembanding sederhana, '
                        'bukan dari model HWD.</div>', unsafe_allow_html=True)

        names = use_cols
        pick = list(range(min(3, K)))
        fig = plot_series(data, pred, ev, cond, 0, pick, names)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # Persamaan (3.14): nilai kondisi dipertahankan, posisi lain diisi
        filled = np.where(cond == 1, data, pred).reshape(-1, K) * sd + mu
        out = df_up.iloc[:n_win * SEQ_LEN].copy()
        out[use_cols] = filled
        rule_cols = {"R703_A", "R705", "R706", "R707"}
        if rule_cols.issubset(df_up.columns) and st.checkbox("Terapkan survey rule engine", value=True):
            asli = prepare_susenas(df_up.iloc[:n_win * SEQ_LEN].reset_index(drop=True))
            out = restore_fn(out.reset_index(drop=True), asli)
            out = out[[c for c in df_up.columns]]
        st.download_button("Unduh hasil imputasi (.csv)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"imputasi_{label.split()[0].lower()}.csv", mime="text/csv",
                           width="stretch")
