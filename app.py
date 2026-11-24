"""Streamlit entry point for the Kelompok 9 hourly CO forecasting dashboard."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from air_data import FEATURES, TARGET, official_csv, read_data
from fuzzy_system import ACTIONS, GRID, IN_CENTERS, LEVELS, OUT_CENTERS, fuzzy_detail, memberships
from inference import DEFAULT, forecast, load_run
from train import MODELS, run

st.set_page_config(page_title="Prakiraan CO | Kelompok 9", layout="wide")
COLORS = ["#18756a", "#5469a2", "#bf702c", "#80579b", "#556775"]
PAGES = ["Ringkasan", "Eksplorasi data", "Seleksi fitur GA", "Optimasi PSO", "Jaringan saraf",
         "Aturan fuzzy", "Prediksi dan simulasi", "Perbandingan model"]
directory = Path(st.session_state.get("active_run", DEFAULT))


@st.cache_data(show_spinner=False)
def read_bundle(path):
    root = Path(path)
    return (load_run(root), pd.read_csv(root / "test_predictions.csv", parse_dates=["timestamp_target"]),
            pd.read_csv(root / "daily.csv", parse_dates=["timestamp"]))


def draw(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def chart(x_label, y_label, height=2.8):
    fig, ax = plt.subplots(figsize=(9.2, height))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    return fig, ax


def plot_histories(logs, xlabel, ylabel):
    fig, ax = chart(xlabel, ylabel)
    for color, log in zip(COLORS, logs):
        h = log["history"]
        ax.plot(range(1, len(h)+1), h, marker="o", markersize=4, color=color, label=f"Seed {log['seed']}")
    ax.legend(frameon=False, ncol=3)
    draw(fig)


def actual_plot(predictions, column, hours):
    subset = predictions.set_index("timestamp_target").sort_index()
    first = subset.index[0]
    subset = subset.loc[:first + pd.Timedelta(hours=hours-1)].asfreq("h")
    fig, ax = chart("Waktu target", "CO (mg/m³)")
    ax.plot(subset.index, subset.aktual, color="#454d5c", label="Aktual", linewidth=1.4)
    ax.plot(subset.index, subset[column], color=COLORS[0], label=MODELS[column], linewidth=1.6)
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    draw(fig)


def measure(value):
    return "—" if pd.isna(value) else f"{value:.3f}"


def results_table(result):
    rows = []
    for key, model in result["models"].items():
        m = model["test"]
        rows.append({"Model": MODELS[key], "Fitur": len(model["features"]), "MAE uji": m["mae"],
                     "RMSE uji": m["rmse"], "R² uji": m["r2"], "RMSE validasi": model["val"]["rmse"],
                     "Latih akhir (s)": model["fit_seconds"]})
    return pd.DataFrame(rows)


try:
    info, predictions, daily = read_bundle(str(directory))
except (OSError, ValueError) as error:
    st.error(f"Artefak eksperimen tidak dapat dimuat. {error}")
    st.stop()

hybrid = info["models"]["hybrid"]
stats = info["data"]
st.sidebar.title("Prakiraan CO")
st.sidebar.caption("Teknik Elektro ITK · Kelompok 9")
page = st.sidebar.radio("Navigasi", PAGES, label_visibility="collapsed")
st.sidebar.caption("Horizon prediksi 1 jam · Satuan mg/m³")
if directory != DEFAULT:
    st.sidebar.info("Eksperimen sesi ini aktif.")
    if st.sidebar.button("Kembali ke eksperimen UCI"):
        st.session_state.pop("active_run", None)
        st.rerun()

if page == PAGES[0]:
    st.title("Prakiraan karbon monoksida")
    st.write("Seberapa tinggi konsentrasi CO satu jam ke depan? Sistem membaca riwayat sensor, "
             "memprakirakan CO dengan RNN, lalu menetapkan prioritas verifikasi pengukuran.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jam dalam dataset", f"{stats['rows']:,}")
    c2.metric("Sensor model hibrida", f"{len(hybrid['features'])} dari {len(FEATURES)}")
    c3.metric("MAE hibrida pada data uji", f"{hybrid['test']['mae']:.3f} mg/m³")
    st.subheader("Aktual dan prakiraan pada 96 jam pertama periode uji")
    actual_plot(predictions, "hybrid", 96)
    st.subheader("Alur sistem")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box]; "Data latih" -> "Seleksi GA" -> "Tuning PSO" -> "Bobot RNN"; "Riwayat sensor" -> "RNN terlatih"; "Bobot RNN" -> "RNN terlatih"; "RNN terlatih" -> "CO t+1" -> "Prioritas fuzzy"; }')
    st.caption("Ariyo Arianto · Jhosua George Kamagi · Vandri Afli Navi Almas · Bernadt Jansen Rajagukguk")
    st.link_button("Sumber data UCI Air Quality", "https://doi.org/10.24432/C59K5F")

elif page == PAGES[1]:
    st.title("Eksplorasi data")
    st.write(f"{stats['rows']:,} jam dari {stats['from']} sampai {stats['to']}. "
             "Tabel menyajikan nilai teramati sebelum imputasi. Kode −200 menandai nilai hilang.")
    summary = pd.DataFrame(stats["summary"]).set_index("variabel")
    table = summary[["observed", "missing", "min", "median", "max"]].copy()
    table["hilang (%)"] = table.missing / stats["rows"] * 100
    st.dataframe(table.round(2), width="stretch")
    var = st.selectbox("Variabel", list(summary.index), index=list(summary.index).index(TARGET))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = chart("Tanggal", f"Rata-rata harian {var}")
        ax.plot(daily.timestamp, daily[var], color=COLORS[0], linewidth=1.2)
        fig.autofmt_xdate()
        draw(fig)
    with c2:
        fig, ax = chart(var, "Jumlah observasi")
        item = summary.loc[var]
        if item["hist"]:
            ax.stairs(item["hist"], item["edges"], fill=True, color=COLORS[1])
        else:
            ax.text(.5, .5, "Tidak ada nilai teramati", ha="center", transform=ax.transAxes)
        draw(fig)
    st.subheader("Pembagian berdasarkan jam target")
    st.dataframe(pd.DataFrame(stats["splits"]).T, width="stretch")
    st.caption("Jam tanpa label CO dikeluarkan. Setiap jendela memakai informasi sebelum jam target. "
               "Median, rata-rata, dan simpangan baku berasal dari data latih.")
    with st.expander("Eksplorasi CSV lain"):
        uploaded = st.file_uploader("CSV UCI atau CSV dengan timestamp ISO", type="csv", key="eda_csv")
        if uploaded is not None:
            try:
                df = read_data(BytesIO(uploaded.getvalue()), [])
                st.dataframe(df.describe().T.assign(hilang=df.isna().sum()), width="stretch")
                st.line_chart(df[st.selectbox("Kolom dari berkas", df.columns)])
            except (ValueError, TypeError, UnicodeError) as error:
                st.error(str(error))

elif page == PAGES[2]:
    st.title("Seleksi fitur dengan GA")
    st.write("Setiap bit mewakili satu kanal. Fitness memakai RMSE validasi ditambah penalti "
             "0,03 × jumlah fitur aktif / 8. Fitness yang lebih kecil lebih baik.")
    plot_histories(info["ga"], "Generasi", "Fitness (mg/m³)")
    st.dataframe([{"Seed": g["seed"], "Fitness": g["fitness"], "Fitur": ", ".join(g["features"]),
                   "Evaluasi": len(g["candidates"]), "Pelatihan baru": g["new_fits"]} for g in info["ga"]],
                 hide_index=True, width="stretch")
    st.write("Fitur yang dipakai model hibrida")
    st.dataframe([{"Kanal": c, "Bit": int(c in hybrid["features"])} for c in FEATURES],
                 hide_index=True, width="stretch")
    st.caption("Enam individu, lima generasi, turnamen tiga individu, crossover satu titik 0,8, "
               "mutasi per bit 1/8, satu individu elit.")

elif page == PAGES[3]:
    st.title("Optimasi hiperparameter dengan PSO")
    choice = st.radio("Ruang fitur", ["Hasil GA", "Seluruh kanal"], horizontal=True)
    logs = info["pso_ga" if choice == "Hasil GA" else "pso_full"]
    plot_histories(logs, "Iterasi", "RMSE validasi terbaik (mg/m³)")
    st.dataframe([{"Seed": g["seed"], **g["parameters"], "RMSE pencarian": g["fitness"]} for g in logs],
                 hide_index=True, width="stretch")
    st.write("Panjang riwayat 4, 6, atau 8 jam. Unit tersembunyi 8, 12, 16, 20, atau 24. "
             "Learning rate 0,001–0,01 pada skala log. Dropout 0–0,25.")
    st.caption("Lima partikel selama lima iterasi per seed. Inersia 0,65 dan kedua koefisien percepatan 1,4. "
               "Posisi terbaik pribadi dan kawanan diperbarui dari galat validasi.")

elif page == PAGES[4]:
    st.title("Jaringan saraf rekuren")
    key = st.selectbox("Model", list(MODELS), format_func=MODELS.get, index=3)
    model = info["models"][key]
    p = model["params"]
    st.write(f"Urutan {p['length']} jam × {len(model['features'])} kanal → RNN Elman dengan "
             f"{p['units']} unit tanh → satu prakiraan CO. Checkpoint terpilih pada epoch {model['best_epoch']}.")
    fig, ax = chart("Epoch", "MSE pada target baku")
    h = pd.DataFrame(model["history"])
    ax.plot(h.epoch, h.train_mse_z, label="Latih", color=COLORS[0])
    ax.plot(h.epoch, h.val_mse_z, label="Validasi", color=COLORS[2])
    ax.axvline(model["best_epoch"], color="#656a73", ls=":", label="Checkpoint")
    ax.legend(frameon=False)
    draw(fig)
    m = model["test"]
    for col, label, val in zip(st.columns(3), ["MAE (mg/m³)", "RMSE (mg/m³)", "R²"], [m["mae"], m["rmse"], m["r2"]]):
        col.metric(label, measure(val))
    actual_plot(predictions, key, 168)

elif page == PAGES[5]:
    st.title("Prioritas pemantauan dengan fuzzy Mamdani")
    st.write("Lima aturan memetakan konsentrasi CO ke prioritas verifikasi. Indeks 0–100 ini "
             "merupakan skala internal proyek untuk pemantauan sensor.")
    c = st.slider("CO untuk memeriksa aturan (mg/m³)", 0., 15., 3., .1)
    score, idx, deg, agg = fuzzy_detail(c)
    left, right = st.columns(2)
    with left:
        fig, ax = chart("CO (mg/m³)", "Derajat keanggotaan")
        x = np.linspace(0, 15, 181)
        for i, vals in enumerate(memberships(x, IN_CENTERS).T):
            ax.plot(x, vals, label=LEVELS[i], color=COLORS[i])
        ax.axvline(c, color="#404650", linestyle=":")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        draw(fig)
    with right:
        fig, ax = chart("Skor prioritas", "Derajat keanggotaan")
        for i, vals in enumerate(memberships(GRID, OUT_CENTERS).T):
            ax.plot(GRID, vals, alpha=.45, color=COLORS[i])
        ax.fill_between(GRID, agg, alpha=.28, color=COLORS[0])
        ax.axvline(score, color=COLORS[0], label=f"Centroid {score:.2f}")
        ax.legend(frameon=False)
        draw(fig)
    st.dataframe([{"Jika CO": label, "Maka prioritas": label, "Aktivasi": round(deg[i], 3),
                   "Tindak lanjut": ACTIONS[i]} for i, label in enumerate(LEVELS)], hide_index=True, width="stretch")
    st.info(f"{score:.2f}/100 · {LEVELS[idx]}. {ACTIONS[idx]}")

elif page == PAGES[6]:
    st.title("Prakiraan satu jam ke depan")
    length = hybrid["params"]["length"]
    st.write(f"Masukkan minimal {length} jam berurutan. Waktu keluaran adalah satu jam setelah baris "
             "terakhir tiap jendela. Angka sensor memakai satuan asli UCI.")
    manual, batch, training = st.tabs(["Input manual", "Unggah CSV", "Pelatihan ulang"])
    with manual:
        sample = pd.read_csv(directory / "contoh_sensor.csv")
        edited = st.data_editor(sample, hide_index=True, width="stretch", key=f"manual_{directory.name}_{info['source_sha256'][:10]}")
        if st.button("Hitung prakiraan", type="primary"):
            try:
                row = forecast(edited, directory).iloc[-1]
                if pd.isna(row.CO_mg_m3):
                    st.warning(row.tindak_lanjut)
                else:
                    a, b = st.columns(2)
                    a.metric("CO (mg/m³)", f"{row.CO_mg_m3:.3f}")
                    b.metric("Skor pemantauan", f"{row.skor:.2f} / 100")
                    st.success(f"{row.timestamp_target} · {row.prioritas}. {row.tindak_lanjut}")
            except (ValueError, TypeError, UnicodeError) as error:
                st.error(str(error))
    with batch:
        st.download_button("Unduh contoh CSV", (directory / "contoh_sensor.csv").read_bytes(),
                           "contoh_sensor.csv", "text/csv")
        csv = st.file_uploader("Riwayat per jam", type="csv")
        if csv is not None:
            try:
                out = forecast(BytesIO(csv.getvalue()), directory)
                st.dataframe(out, hide_index=True, width="stretch")
                st.download_button("Unduh hasil prediksi", out.to_csv(index=False), "prakiraan_CO.csv", "text/csv")
            except (ValueError, TypeError, UnicodeError) as error:
                st.error(str(error))
    with training:
        st.write("Jalankan empat model kembali dengan parameter eksperimen yang sama. Hasil berlaku untuk sesi ini.")
        training_csv = st.file_uploader("Data latih alternatif (opsional)", type="csv")
        if st.button("Latih empat model"):
            try:
                with st.spinner("Melatih model dan memeriksa checkpoint..."):
                    raw = training_csv.getvalue() if training_csv else official_csv(Path(".cache") / "AirQualityUCI.csv")
                    data = read_data(BytesIO(raw), [*FEATURES, TARGET])
                    temp = TemporaryDirectory(prefix="co-run-")
                    active = run(data, Path(temp.name) / "session", sha256(raw).hexdigest())
                    st.session_state["run_temp"] = temp
                    st.session_state["active_run"] = str(active)
                st.rerun()
            except (OSError, ValueError, TypeError, UnicodeError) as error:
                st.error(str(error))

else:
    st.title("Perbandingan empat model")
    st.write(f"Keempat konfigurasi diuji pada {len(predictions):,} waktu target yang identik. "
             "RMSE memberi penalti lebih besar pada kesalahan besar. MAE menunjukkan besar galat rata-rata.")
    table = results_table(info)
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    perf = info["persistence"]
    st.write(f"Pembanding persistensi menggunakan CO teramati terakhir. MAE {perf['mae']:.3f} mg/m³, "
             f"RMSE {perf['rmse']:.3f} mg/m³ dan R² {measure(perf['r2'])}.")
    key = st.selectbox("Kurva prediksi", list(MODELS), format_func=MODELS.get, index=3)
    actual_plot(predictions, key, st.slider("Rentang jam sejak awal uji", 24, 720, 168, 24))
    st.download_button("Unduh semua prediksi uji", (directory / "test_predictions.csv").read_bytes(),
                       "prediksi_uji.csv", "text/csv")
