"""Eight-page Streamlit dashboard for the TE2514031 air-quality project."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

from air_data import FEATURES, TARGET, official_csv, read_data
from evaluation import MODEL_LABELS as MODELS, comparison, error_groups
from fuzzy_system import ACTIONS, GRID, IN_CENTERS, LEVELS, OUT_CENTERS, fuzzy_detail, memberships
from inference import DEFAULT, forecast, load_run, sample_input, verify_bundle
from train import run
from dashboard_ui import (COLORS, GRID_COLOR, INK, PAPER, init_style, sidebar_brand, sidebar_status,
                          hero, page_header, note, method_grid, flow_grid, show_altair, footer)

st.set_page_config(page_title="Air Quality Lab | Kelompok 9", page_icon="🌿", layout="wide",
                   initial_sidebar_state="auto")
PAGES = ["Ringkasan", "Eksplorasi data", "Seleksi fitur GA", "Optimasi PSO", "Jaringan saraf",
         "Aturan fuzzy", "Prediksi dan simulasi", "Perbandingan model"]
SUBTITLES = ["Overview", "Dataset Explorer", "GA Optimization", "PSO Optimization",
             "Neural Network", "Fuzzy Inference", "Prediction / Simulation", "Model Comparison"]
MEMBERS = [("Ariyo Arianto", "04231013"), ("Jhosua George Kamagi", "04231043"),
           ("Vandri Afli Navi Almas", "04231089"), ("Bernadt Jansen Rajagukguk", "04221062")]

init_style()

directory = Path(st.session_state.get("active_run", DEFAULT))


@st.cache_data(show_spinner=False)
def read_bundle(path, manifest_hash):
    root = Path(path)
    return (load_run(root), pd.read_csv(root / "test_predictions.csv", parse_dates=["timestamp_target"]),
            pd.read_csv(root / "daily.csv", parse_dates=["timestamp"]),
            pd.read_csv(root / "correlation.csv", index_col=0))


def draw(fig):
    fig.tight_layout(pad=1.15)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def chart(xlabel, ylabel, height=3.0, width=9.2):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#BFCFD5")
    ax.tick_params(labelsize=8, colors="#4B626E")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=.7)
    ax.set_axisbelow(True)
    return fig, ax


def metric_row(items):
    for col, item in zip(st.columns(len(items)), items):
        label, value, *rest = item
        col.metric(label, value, border=True, help=rest[0] if rest else None)


def page_title(index, title, description=None):
    page_header(index, SUBTITLES[index], title, description)


def plot_histories(logs, xlabel, ylabel):
    series = pd.DataFrame([{"Tahap": step + 1, "Galat": value, "Seed": f"Seed {log['seed']}"}
                           for log in logs for step, value in enumerate(log["history"])])
    encoding = alt.Chart(series).encode(
        x=alt.X("Tahap:O", title=xlabel, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Galat:Q", title=ylabel, scale=alt.Scale(zero=False)),
        color=alt.Color("Seed:N", scale=alt.Scale(domain=[f"Seed {log['seed']}" for log in logs],
                                                  range=COLORS[:len(logs)]), title=None),
        tooltip=[alt.Tooltip("Seed:N"), alt.Tooltip("Tahap:O", title=xlabel),
                 alt.Tooltip("Galat:Q", title=ylabel, format=".5f")])
    show_altair(encoding.mark_line(strokeWidth=2.4, point=True), 245)


def actual_plot(predictions, column, hours, offset=0):
    series = predictions.set_index("timestamp_target").sort_index()
    start = series.index[0] + pd.Timedelta(hours=offset)
    stop = start + pd.Timedelta(hours=hours-1)
    subset = series.loc[start:stop, ["aktual", column]].asfreq("h").reset_index()
    if subset.empty:
        st.info("Tidak ada target valid pada rentang ini. Pilih awal periode lain.")
        return
    names = {"aktual": "Aktual", column: MODELS[column]}
    frame = subset.melt("timestamp_target", var_name="Seri", value_name="CO")
    # Keep discontinuities in each series when target observations were excluded.
    frame["Segmen"] = frame.groupby("Seri")["CO"].transform(lambda c: c.isna().cumsum()).astype(str)
    frame["Seri"] = frame["Seri"].map(names)
    frame = frame.dropna(subset=["CO"])
    base = alt.Chart(frame).encode(
        x=alt.X("timestamp_target:T", title="Waktu target", axis=alt.Axis(format="%d %b %H:%M", tickCount=6, labelAngle=-20)),
        y=alt.Y("CO:Q", title="Konsentrasi CO (mg/m³)", scale=alt.Scale(zero=True)),
        color=alt.Color("Seri:N", title=None, legend=alt.Legend(symbolType="stroke", symbolOpacity=1), scale=alt.Scale(
            domain=["Aktual", MODELS[column]], range=["#8398AB", COLORS[0]])),
        detail="Segmen:N",
        tooltip=[alt.Tooltip("timestamp_target:T", title="Jam target", format="%d %b %Y, %H:%M"),
                 alt.Tooltip("Seri:N"), alt.Tooltip("CO:Q", title="CO (mg/m³)", format=".3f")])
    show_altair(base.mark_line(strokeWidth=2.2) + base.mark_circle(size=52, opacity=.09), 262)


def measure(value, places=3):
    return "—" if pd.isna(value) else f"{value:.{places}f}"


def statistics(logs):
    a = np.array([g["fitness"] for g in logs])
    return f"{a.mean():.5f}", f"{a.std(ddof=1):.5f}" if len(a) > 1 else "—"


def fit_notes():
    st.caption(f"Pencarian: {info['search_train']:,} jendela latih, {info['search_epochs']} epoch, "
               f"seed model {info['search_init_seed']}. Empat pelatihan akhir menggunakan semua "
               f"{stats['splits']['train']['n']:,} jendela latih; seed {info['final_init_seed']}. "
               "Kandidat dipilih pada validasi; data uji hanya untuk evaluasi akhir.")


# Check the manifest on every new render; cached tables are keyed by its hash.
try:
    manifest = verify_bundle(directory)
    bundle_hash = sha256((directory / "bundle_manifest.json").read_bytes()).hexdigest()
    info, predictions, daily, corr = read_bundle(str(directory), bundle_hash)
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Artefak eksperimen tidak dapat dimuat. {error}")
    st.stop()

hybrid = info["models"]["hybrid"]
stats = info["data"]
model_table = comparison(info)
best_mae = model_table.loc[model_table["MAE uji"].idxmin()]
best_rmse = model_table.loc[model_table["RMSE uji"].idxmin()]
if "R² uji" in model_table and model_table["R² uji"].notna().any():
    best_r2 = model_table.loc[model_table["R² uji"].idxmax()]
else:
    best_r2 = None

sidebar_brand()
page = st.sidebar.radio("Navigasi", PAGES, label_visibility="collapsed", key="nav")
st.sidebar.divider()
sidebar_status(info["source_kind"], stats["rows"])
st.sidebar.link_button("Dataset UCI · Sumber", "https://doi.org/10.24432/C59K5F", width="stretch")
st.sidebar.link_button("Source code · GitHub", "https://github.com/04231043/tubes-kecebut-kel9", width="stretch")


def navigate(target):
    st.session_state["nav"] = target


if directory != DEFAULT:
    st.sidebar.warning("Eksperimen sesi ini aktif.")
    if st.sidebar.button("Kembali ke eksperimen UCI"):
        st.session_state.pop("active_run", None)
        st.session_state.pop("run_temp", None)
        st.rerun()


if page == PAGES[0]:
    hero()
    cta, compare_btn = st.columns(2)
    cta.button("Mulai simulasi prakiraan", type="primary", width="stretch",
               on_click=navigate, args=(PAGES[6],))
    compare_btn.button("Lihat perbandingan model", width="stretch",
                       on_click=navigate, args=(PAGES[7],))
    h = hybrid["test"]
    metric_row([("Observasi per jam", f"{stats['rows']:,}", "Data pada periode historis."),
                ("Kanal hibrida", f"{len(hybrid['features'])} dari {len(FEATURES)}", "Kanal hasil seleksi GA."),
                ("MAE hibrida", f"{h['mae']:.3f} mg/m³", "Rata-rata galat absolut pada data uji."),
                ("RMSE hibrida", f"{h['rmse']:.3f} mg/m³", "Akar rata-rata kuadrat galat.")])
    left, right = st.columns([2.1, 1])
    with left:
        with st.container(border=True):
            st.subheader("Konsentrasi CO · Aktual vs prakiraan")
            hours = st.radio("Rentang observasi", [48, 96, 168], index=1,
                             format_func=lambda n: f"{n} jam", horizontal=True, label_visibility="collapsed")
            actual_plot(predictions, "hybrid", hours)
            st.caption("Arahkan kursor ke grafik untuk melihat waktu dan nilai. Celah menandakan target yang tidak lolos filter.")
    with right:
        with st.container(border=True):
            st.subheader("Ringkasan hasil uji")
            st.caption(f"{len(predictions):,} waktu target yang sama")
            st.markdown(f"**MAE terendah**  \n{best_mae['Model']} · {best_mae['MAE uji']:.4f}")
            st.markdown(f"**RMSE terendah**  \n{best_rmse['Model']} · {best_rmse['RMSE uji']:.4f}")
            if best_r2 is not None:
                st.markdown(f"**R² tertinggi**  \n{best_r2['Model']} · {best_r2['R² uji']:.4f}")
            note("Interpretasi", "Model terbaik bergantung pada metrik. Fuzzy hanya menetapkan prioritas pengecekan, bukan menaikkan akurasi RNN.")
    st.subheader("Empat modul kecerdasan buatan")
    method_grid([
        ("01 · OPTIMASI", "Genetic Algorithm", "Kromosom biner memilih kanal sensor dengan fitness validasi.",
         f"{len(hybrid['features'])} dari {len(FEATURES)} kanal terpilih"),
        ("02 · OPTIMASI", "Particle Swarm", "Pencarian panjang jendela, unit, learning rate, dan dropout.",
         f"{hybrid['params']['length']} jam · {hybrid['params']['units']} unit"),
        ("03 · PREDIKSI", "Recurrent NN", "RNN Elman membaca urutan sensor untuk CO pada jam berikutnya.",
         f"MAE {h['mae']:.4f} mg/m³"),
        ("04 · KEPUTUSAN", "Fuzzy Mamdani", "Aturan min–max dan centroid menentukan prioritas pemeriksaan.",
         "5 aturan · skor 0–100")
    ])
    st.subheader("Bagaimana sistem bekerja?")
    with st.container(border=True):
        st.markdown("**Jalur pelatihan (offline)**")
        flow_grid([("Langkah 01", "Data & preprocessing", "Split waktu 70/15/15; statistik transformasi hanya dari latih."),
                   ("Langkah 02", "GA + PSO", "Subset sensor dan hyperparameter dipilih memakai validasi."),
                   ("Langkah 03", "Empat checkpoint", "Bobot RNN dibekukan sebelum evaluasi pada data uji.")])
        st.markdown("**Jalur pemakaian (inference)**")
        flow_grid([("Input", "Riwayat sensor sampai t", "Sertakan kanal terpilih dan timestamp berurutan."),
                   ("Forecast", "CO pada t+1", "Standardisasi dan RNN terlatih menghasilkan mg/m³."),
                   ("Output", "Keputusan fuzzy", "Tindak lanjut verifikasi sensor pada skala internal proyek.")])
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.subheader("Data dan pembagian eksperimen")
            split_rows = pd.DataFrame([{"Bagian": label, "Jam target valid": stats["splits"][key]["n"]}
                                       for key, label in [("train", "Latih"), ("val", "Validasi"), ("test", "Uji")]])
            split_chart = alt.Chart(split_rows).mark_bar(cornerRadiusEnd=5, size=22).encode(
                y=alt.Y("Bagian:N", sort=["Latih", "Validasi", "Uji"], title=None),
                x=alt.X("Jam target valid:Q", title="Jumlah target valid"),
                color=alt.Color("Bagian:N", scale=alt.Scale(domain=["Latih", "Validasi", "Uji"],
                                                             range=COLORS[:3]), legend=None),
                tooltip=["Bagian:N", "Jam target valid:Q"])
            show_altair(split_chart, 130)
            st.caption("Pembagian 70/15/15 berdasarkan periode waktu sebelum filter target dan jendela.")
    with b:
        with st.container(border=True):
            st.subheader("Tim proyek · Kelompok 9")
            st.caption("Teknik Elektro · Institut Teknologi Kalimantan · Kelas B")
            st.dataframe(pd.DataFrame(MEMBERS, columns=["Nama", "NIM"]), hide_index=True, width="stretch")
    with st.expander("Bukti keterulangan & panduan demo"):
        fit_notes()
        st.code(f"SHA-256 CSV: {info['source_sha256']}\nBundle: {bundle_hash}", language=None)
        st.download_button("Unduh log eksperimen JSON", (directory / "experiment.json").read_bytes(),
                           "experiment.json", "application/json")
        st.write("Mulai dari EDA → GA → PSO → RNN → Fuzzy → Prediksi → Perbandingan. "
                 "Model terlatih dapat digunakan tanpa pelatihan ulang.")


elif page == PAGES[1]:
    page_title(1, "Eksplorasi data", "Observasi historis sebelum imputasi · kode −200 dianggap data hilang")
    summary = pd.DataFrame(stats["summary"]).set_index("variabel")
    table = summary[["observed", "missing", "min", "median", "max"]].copy()
    table["hilang (%)"] = table.missing / stats["rows"] * 100
    metric_row([("Jam tercatat", f"{stats['rows']:,}"),
                ("CO teramati", f"{int(summary.loc[TARGET, 'observed']):,}"),
                ("CO hilang", f"{table.loc[TARGET, 'hilang (%)']:.1f}%"),
                ("Kandidat masukan", len(FEATURES))])
    st.caption(f"Periode {stats['from']} — {stats['to']} (timestamp lokal sumber).")
    note("Pahami kualitas data terlebih dahulu",
         "Kode −200 bukan konsentrasi negatif, melainkan pembacaan hilang. Target CO yang kosong tidak dilatih; "
         "kanal masukan diimputasi dengan median data latih.")
    left, right = st.columns([1.2, 1])
    with left:
        with st.container(border=True):
            st.subheader("Distribusi data hilang")
            selected = table.reset_index().rename(columns={"variabel": "Kanal", "hilang (%)": "Persen hilang"})
            missing_bars = alt.Chart(selected).mark_bar(cornerRadiusEnd=4, size=15).encode(
                y=alt.Y("Kanal:N", title=None, sort="-x", axis=alt.Axis(labelLimit=140)),
                x=alt.X("Persen hilang:Q", title="Nilai hilang (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.condition(alt.datum.Kanal == TARGET, alt.value(COLORS[2]), alt.value(COLORS[0])),
                tooltip=["Kanal:N", alt.Tooltip("missing:Q", title="Jumlah hilang"),
                         alt.Tooltip("Persen hilang:Q", format=".2f"), alt.Tooltip("observed:Q", title="Teramati")])
            show_altair(missing_bars, 300)
            st.caption("Warna jingga menunjukkan label target CO; arahkan kursor ke grafik untuk melihat jumlah pembacaan.")
    with right:
        with st.container(border=True):
            st.subheader("Korelasi observasi")
            labels = {x: "CO" if x == TARGET else x.split("(")[0].replace("PT08.", "") for x in corr.columns}
            frame = corr.rename(index=labels, columns=labels).rename_axis("Y").reset_index().melt("Y", var_name="X", value_name="r")
            order = list(labels.values())
            base = alt.Chart(frame).encode(x=alt.X("X:O", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Y:O", title=None, sort=order), tooltip=["X:O", "Y:O", alt.Tooltip("r:Q", title="Pearson r", format=".3f")])
            cells = base.mark_rect(stroke="white", strokeWidth=1).encode(color=alt.Color(
                "r:Q", title="Pearson r", scale=alt.Scale(domain=[-1, 0, 1], range=["#5074B8", "#F4F6F8", "#C56B45"]), legend=alt.Legend(orient="right")))
            nums = base.mark_text(fontSize=10).encode(text=alt.Text("r:Q", format=".1f"), color=alt.value(INK))
            show_altair(cells + nums, 290)
            st.caption("Pearson pada pasangan nilai teramati; hanya untuk EDA, bukan untuk memilih fitur.")
    st.subheader("Statistik deskriptif")
    st.dataframe(table.rename(columns={"observed": "Teramati", "missing": "Hilang",
                                       "min": "Min", "median": "Median", "max": "Maks"}).round(2), width="stretch")
    variable = st.selectbox("Variabel untuk melihat tren", list(summary.index), index=list(summary.index).index(TARGET))
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Tren harian")
            series = daily[["timestamp", variable]].rename(columns={variable: "nilai"})
            trend = alt.Chart(series).mark_line(color=COLORS[0], strokeWidth=2).encode(
                x=alt.X("timestamp:T", title="Tanggal", axis=alt.Axis(format="%b %Y", tickCount=5)),
                y=alt.Y("nilai:Q", title=f"Rata-rata {variable}", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("timestamp:T", title="Tanggal", format="%d %b %Y"),
                         alt.Tooltip("nilai:Q", title=variable, format=".3f")])
            show_altair(trend, 220)
    with c2:
        with st.container(border=True):
            st.subheader("Distribusi nilai")
            fig, ax = chart(variable, "Jumlah observasi", width=7)
            item = summary.loc[variable]
            if item["hist"]:
                ax.stairs(item["hist"], item["edges"], fill=True, color=COLORS[1], alpha=.85)
            else:
                ax.text(.5, .5, "Tidak ada nilai teramati", ha="center", transform=ax.transAxes)
            draw(fig)
    st.subheader("Split kronologis 70 / 15 / 15")
    split = pd.DataFrame(stats["splits"]).T
    st.bar_chart(split[["hours", "n"]].rename(columns={"hours": "Jam periode", "n": "Target valid"}), horizontal=True)
    st.dataframe(split.rename(columns={"from": "Awal", "to": "Akhir", "hours": "Jam periode",
                                       "observed_target": "CO teramati", "n": "Target valid"}), width="stretch")
    flow_grid([("70%", "Latih", "Pelajari median, scaler, dan bobot RNN."),
               ("15%", "Validasi", "Pilih fitur, hyperparameter dan checkpoint."),
               ("15%", "Uji", "Evaluasi akhir tanpa penyesuaian model.")])
    note("Tidak ada kebocoran target", "Jendela sensor berhenti pada t; CO diprakirakan pada t+1. "
         "Semua konfigurasi menggunakan waktu target uji yang sama.", "good")
    with st.expander("Kamus data dan sumber"):
        labels = [
            ("CO(GT)", "Target referensi", "mg/m³"),
            *[(feature, "Respons sensor kimia; kandidat masukan", "Nilai asli UCI") for feature in FEATURES[:5]],
            ("T", "Temperatur", "°C"), ("RH", "Kelembapan relatif", "%"),
            ("AH", "Kelembapan absolut", "Nilai asli UCI"),
            ("timestamp", "Waktu observasi lokal", "Jam penuh")
        ]
        st.dataframe(pd.DataFrame(labels, columns=["Kolom", "Peran", "Satuan / format"]), hide_index=True, width="stretch")
        st.write("Kanal konsentrasi gas referensi lainnya tersedia untuk EDA, tetapi tidak digunakan sebagai fitur.")
    with st.expander("Eksplorasi CSV lain (tidak mengubah model)"):
        uploaded = st.file_uploader("CSV UCI atau CSV dengan timestamp ISO", type="csv", key="eda_csv")
        if uploaded is not None:
            try:
                df = read_data(BytesIO(uploaded.getvalue()), [])
                if not len(df.columns):
                    raise ValueError("CSV tidak memuat kolom pengukuran.")
                st.dataframe(df.describe().T.assign(hilang=df.isna().sum()), width="stretch")
                st.line_chart(df[st.selectbox("Kolom CSV", df.columns)])
            except (ValueError, TypeError, UnicodeError) as error:
                st.error(str(error))


elif page == PAGES[2]:
    page_title(2, "Genetic Algorithm", "Seleksi fitur dengan kromosom biner dan validasi terpisah")
    logs = info["ga"]
    selected = hybrid["features"]
    best = min(logs, key=lambda g: g["fitness"])
    mean, sd = statistics(logs)
    chromosome = "".join("1" if c in selected else "0" for c in FEATURES)
    metric_row([("Fitness terbaik", measure(best["fitness"], 5)),
                ("Fitur terpilih", f"{len(selected)} / {len(FEATURES)}"),
                ("Seed terpilih", best["seed"]), ("Pencarian baru", sum(g["new_fits"] for g in logs))])
    st.latex(r"J(\mathbf{b})=\operatorname{RMSE}_{val}+\lambda \frac{\sum b_i}{8}")
    st.caption(f"Minimasi fitness · penalti λ={info['ga_penalty']:.2f} mg/m³ · kromosom terpilih {chromosome}")
    flow_grid([("01 · Encoding", "8 bit kandidat", "Bit 1 mengaktifkan kanal; bit 0 mengeluarkannya."),
               ("02 · Evolusi", "Fitness, seleksi, variasi", "Turnamen, crossover dan mutasi membentuk kandidat baru."),
               ("03 · Hasil", "Kromosom terbaik", "Elitisme mempertahankan fitness terbaik antargenerasi.")])
    left, right = st.columns([1.45, 1])
    with left:
        with st.container(border=True):
            st.subheader("Konvergensi setiap seed")
            plot_histories(logs, "Generasi", "Fitness (mg/m³)")
            st.caption("Semakin kecil semakin baik; nilai terbaik dipertahankan melalui elitisme.")
    with right:
        with st.container(border=True):
            st.subheader("Kromosom terpilih")
            st.code(chromosome, language=None)
            st.dataframe(pd.DataFrame([{"Kanal": c, "Bit": int(c in selected),
                                        "Status": "Dipakai" if c in selected else "Tidak dipakai"} for c in FEATURES]),
                         hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([{"Seed": g["seed"], "Fitness": g["fitness"], "Fitur": ", ".join(g["features"]),
                               "Evaluasi": len(g["candidates"]), "Pelatihan baru": g["new_fits"],
                               "Waktu (s)": g["seconds"]} for g in logs]).round(5), hide_index=True, width="stretch")
    st.caption(f"Fitness akhir (antar-seed): rata-rata {mean}, simpangan baku sampel {sd}. "
               "Turnamen 3 individu, crossover 0,8, mutasi per bit 1/8, elitisme 1; "
               f"{info['search_budget']['ga_population']} individu × {info['search_budget']['ga_generations']} generasi per seed.")
    st.subheader("Efek seleksi fitur pada model dasar")
    st.dataframe(model_table.iloc[:2][["Model", "Fitur", "RMSE validasi", "MAE uji", "RMSE uji", "Latih akhir (s)"]].round(4),
                 hide_index=True, width="stretch")
    reduction = 100 * (1 - len(selected) / len(FEATURES))
    note("Interpretasi seleksi fitur", f"Dimensi masukan berkurang {reduction:.0f}%. Penalti fitness menyeimbangkan "
         "jumlah kanal dengan galat validasi. RMSE data uji tetap dilaporkan terpisah; pengurangan fitur tidak menjamin semua metrik membaik.")
    fit_notes()
    with st.expander("Log setiap evaluasi kandidat"):
        df = pd.DataFrame([{"seed": g["seed"], **c} for g in logs for c in g["candidates"]])
        st.dataframe(df.round(6), hide_index=True, width="stretch")
        st.download_button("Unduh log GA", df.to_csv(index=False), "log_ga.csv", "text/csv")


elif page == PAGES[3]:
    page_title(3, "Particle Swarm Optimization", "Tuning panjang jendela, unit tersembunyi, learning rate dan dropout")
    choice = st.radio("Ruang fitur", ["Hasil GA", "Seluruh kanal"], horizontal=True)
    logs = info["pso_ga" if choice == "Hasil GA" else "pso_full"]
    key = "hybrid" if choice == "Hasil GA" else "pso"
    best = min(logs, key=lambda g: g["fitness"])
    params = best["parameters"]
    mean, sd = statistics(logs)
    metric_row([("RMSE pencarian", measure(best["fitness"], 5)),
                ("Jendela terbaik", f"{params['length']} jam"), ("Unit RNN", params["units"]) ,
                ("Learning rate", f"{params['lr']:.5f}")])
    st.latex(r"v_i^{t+1}=0.65v_i^t+1.4r_1(pbest_i-x_i^t)+1.4r_2(gbest-x_i^t)")
    st.latex(r"x_i^{t+1}=\operatorname{clip}(x_i^t+v_i^{t+1},0,1)")
    flow_grid([("01 · Posisi", "Kandidat hyperparameter", "Posisi partikel didekode menjadi konfigurasi RNN."),
               ("02 · Evaluasi", "RMSE validasi", "Setiap kandidat melatih RNN pilot dan mengevaluasi validasi."),
               ("03 · Pembaruan", "pbest dan gbest", "Kecepatan diperbarui dari pengalaman pribadi dan kawanan.")])
    left, right = st.columns([1.5, 1])
    with left:
        with st.container(border=True):
            st.subheader("Konvergensi gbest")
            plot_histories(logs, "Iterasi", "RMSE validasi terbaik (mg/m³)")
            st.caption("Garis adalah RMSE terbaik kumulatif dari tiap kawanan, bukan kurva loss epoch RNN.")
    with right:
        with st.container(border=True):
            st.subheader("Ruang pencarian")
            st.dataframe(pd.DataFrame([
                ("Jendela", "4, 6, 8 jam", str(params["length"])),
                ("Unit", "8, 12, 16, 20, 24", str(params["units"])),
                ("Learning rate", "0,001 – 0,01 (log)", str(params["lr"])),
                ("Dropout", "0 – 0,25", str(params["dropout"]))
            ], columns=["Parameter", "Batas", "Terpilih"]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([{"Seed": g["seed"], **g["parameters"], "RMSE pencarian": g["fitness"],
                               "Evaluasi": len(g["candidates"]), "Waktu (s)": g["seconds"]} for g in logs]).round(5),
                 hide_index=True, width="stretch")
    st.caption(f"{info['search_budget']['pso_particles']} partikel × {info['search_budget']['pso_iterations']} iterasi/seed. "
               f"Rata-rata gbest akhir {mean}; simpangan baku sampel {sd}. "
               f"Model akhir {MODELS[key]} memiliki RMSE validasi {info['models'][key]['val']['rmse']:.4f} mg/m³.")
    note("Pencarian versus pelatihan akhir", "PSO memakai anggaran pelatihan pilot. Konfigurasi terpilih kemudian dilatih "
         "dengan seluruh jendela latih dan checkpoint validasi. Karena itu, galat pencarian tidak identik dengan galat akhir.")
    fit_notes()
    with st.expander("Riwayat posisi partikel dan uji kandidat"):
        df = pd.DataFrame([{"seed": g["seed"], "iterasi": c["iteration"], "partikel": c["particle"],
                            **c["params"], "RMSE validasi": c["val_rmse"]} for g in logs for c in g["candidates"]])
        st.dataframe(df.round(6), hide_index=True, width="stretch")
        st.download_button("Unduh log PSO", df.to_csv(index=False), f"log_pso_{key}.csv", "text/csv")


elif page == PAGES[4]:
    page_title(4, "Recurrent Neural Network", "RNN Elman · Backpropagation Through Time · Adam")
    key = st.selectbox("Model", list(MODELS), format_func=MODELS.get, index=3)
    model = info["models"][key]
    p = model["params"]
    t = model["test"]
    metric_row([("MAE (mg/m³)", measure(t["mae"], 4)), ("RMSE (mg/m³)", measure(t["rmse"], 4)),
                ("R²", measure(t["r2"], 4)), ("Checkpoint", f"Epoch {model['best_epoch']}")])
    st.markdown(f"**Arsitektur:** {p['length']} jam × {len(model['features'])} kanal → "
                f"{p['units']} unit tanh → 1 nilai CO. "
                f"Parameter trainable: {model['n_weights']:,}; dropout: {p['dropout']:.2f}.")
    flow_grid([("Input", f"{p['length']} × {len(model['features'])}", "Jendela riwayat semua kanal terpilih, sudah distandardisasi."),
               ("Hidden state", f"{p['units']} unit tanh", "State rekuren membawa informasi dari observasi sebelumnya."),
               ("Output", "1 nilai CO (t+1)", "Lapisan linear, inverse scaling, dan pembatasan nonnegatif.")])
    st.latex(r"h_t=\tanh(W_x x_t+W_h h_{t-1}+b_h),\qquad \widehat{y}_{t+1}=W_y h_t+b_y")
    h = pd.DataFrame(model["history"])
    with st.container(border=True):
        st.subheader("Kurva pembelajaran")
        losses = h.melt("epoch", var_name="Split", value_name="MSE")
        losses["Split"] = losses.Split.map({"train_mse_z": "Latih", "val_mse_z": "Validasi"})
        curve = alt.Chart(losses).mark_line(strokeWidth=2.4, point=True).encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("MSE:Q", title="MSE target baku", scale=alt.Scale(zero=False)),
            color=alt.Color("Split:N", title=None, scale=alt.Scale(domain=["Latih", "Validasi"],
                                                                   range=[COLORS[0], COLORS[2]])),
            tooltip=["Split:N", "epoch:Q", alt.Tooltip("MSE:Q", format=".5f")])
        ckpt = alt.Chart(pd.DataFrame({"epoch": [model["best_epoch"]]})).mark_rule(
            color="#718797", strokeDash=[5, 4], size=1.7).encode(x="epoch:Q")
        show_altair(curve + ckpt, 260)
        st.caption(f"Checkpoint pada epoch {model['best_epoch']}. MSE ini memakai target yang distandardisasi, "
                   f"sedangkan RMSE validasi {model['val']['rmse']:.4f} mg/m³ sudah dalam satuan asli.")
    with st.container(border=True):
        st.subheader("Aktual dan prediksi")
        actual_plot(predictions, key, 168)
    a, b = st.columns(2)
    residual = predictions[key] - predictions.aktual
    with a:
        fig, ax = chart("CO aktual (mg/m³)", "CO prediksi (mg/m³)", 3.4, 5.4)
        ax.scatter(predictions.aktual, predictions[key], color=COLORS[0], alpha=.32, s=10, edgecolors="none")
        maximum = max(predictions.aktual.max(), predictions[key].max())
        ax.plot([0, maximum], [0, maximum], color=COLORS[2], ls="--", linewidth=1.3, label="Ideal")
        ax.legend(frameon=False)
        draw(fig)
    with b:
        fig, ax = chart("Prediksi − aktual (mg/m³)", "Jumlah target", 3.4, 5.4)
        ax.hist(residual, bins=26, color=COLORS[1], alpha=.85)
        ax.axvline(0, color=INK, ls="--")
        draw(fig)
    st.caption("Kurva dan scatter memakai checkpoint; galat positif = prediksi terlalu tinggi. "
               f"Bias = {t['bias']:.4f} mg/m³.")
    fit_notes()
    st.download_button("Unduh kurva pelatihan", h.to_csv(index=False), f"history_{key}.csv", "text/csv")


elif page == PAGES[5]:
    page_title(5, "Fuzzy Inference System", "Mamdani min–max dan centroid: prioritas verifikasi sensor")
    st.warning("Skor 0–100 adalah indeks internal proyek untuk pemantauan sensor, bukan AQI/ISPU resmi, "
               "batas paparan kesehatan, atau rekomendasi keselamatan.")
    flow_grid([("01 · Fuzzifikasi", "CO menjadi derajat μ", "Fungsi keanggotaan mengevaluasi input konsentrasi."),
               ("02 · Inferensi", "Min dan max", "Potong keluaran dengan min dan gabungkan semua aturan dengan max."),
               ("03 · Defuzzifikasi", "Skor centroid", "Hitung titik berat area agregasi pada skala 0–100.")])
    c = st.slider("Konsentrasi CO contoh (mg/m³)", 0., 15., 3., .1)
    score, idx, deg, agg = fuzzy_detail(c)
    metric_row([("CO masukan", f"{c:.1f} mg/m³"), ("Centroid", f"{score:.2f} / 100"), ("Prioritas", LEVELS[idx])])
    left, right = st.columns(2)
    with left:
        st.subheader("Fungsi keanggotaan input")
        fig, ax = chart("CO (mg/m³)", "Derajat keanggotaan", 3.3, 6)
        x = np.linspace(0, 15, 181)
        for i, vals in enumerate(memberships(x, IN_CENTERS).T):
            ax.plot(x, vals, label=LEVELS[i], color=COLORS[i], linewidth=1.6)
        ax.axvline(c, color=INK, linestyle=":")
        ax.set_ylim(-.02, 1.07)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        draw(fig)
    with right:
        st.subheader("Agregasi keluaran dan centroid")
        fig, ax = chart("Skor prioritas", "Derajat keanggotaan", 3.3, 6)
        for i, vals in enumerate(memberships(GRID, OUT_CENTERS).T):
            ax.plot(GRID, vals, alpha=.6, color=COLORS[i])
        ax.fill_between(GRID, agg, alpha=.33, color=COLORS[0], label="Agregasi max")
        ax.axvline(score, color=COLORS[0], label=f"Centroid {score:.2f}")
        ax.legend(frameon=False, fontsize=8)
        draw(fig)
    st.latex(r"s^*=\frac{\int_0^{100}s\,\mu_{agg}(s)\,ds}{\int_0^{100}\mu_{agg}(s)\,ds}")
    st.subheader("Aktivasi rule base")
    st.dataframe([{"Aturan": f"R{i+1}", "Jika CO": label, "Maka prioritas": label,
                   "Aktivasi": round(deg[i], 3), "Tindak lanjut": ACTIONS[i]} for i, label in enumerate(LEVELS)],
                 hide_index=True, width="stretch",
                 column_config={"Aktivasi": st.column_config.ProgressColumn("Aktivasi μ", min_value=0., max_value=1., format="%.3f")})
    st.info(f"{LEVELS[idx]} · {ACTIONS[idx]}")
    st.subheader("Pengujian skenario")
    cases = []
    for value in [0, 2, 3, 4, 7, 10, 15]:
        s, lev, _, _ = fuzzy_detail(value)
        cases.append({"CO (mg/m³)": value, "Skor": round(s, 2), "Prioritas": LEVELS[lev]})
    st.dataframe(cases, hide_index=True, width="stretch")
    st.caption("Himpunan linier dengan bahu di ujung; input di atas 10 mg/m³ tetap memasuki himpunan tertinggi. "
               "Fuzzy tidak mengubah nilai prakiraan RNN.")


elif page == PAGES[6]:
    page_title(6, "Prediksi dan simulasi", "Inference dari checkpoint tanpa pelatihan ulang")
    key = st.selectbox("Konfigurasi model", list(MODELS), format_func=MODELS.get, index=3)
    spec = info["models"][key]
    length, cols = spec["params"]["length"], spec["features"]
    note("Model aktif", f"{MODELS[key]} · {length} jam riwayat × {len(cols)} kanal · "
         f"MAE historis {spec['test']['mae']:.4f} mg/m³. Hasil diberikan pada satu jam setelah observasi terakhir.")
    flow_grid([("Langkah 01", "Input riwayat", "Gunakan contoh atau unggah CSV sensor dengan timestamp jam penuh."),
               ("Langkah 02", "Jalankan inference", "Model terlatih membaca input; tidak perlu proses GA/PSO ulang."),
               ("Langkah 03", "Interpretasi & unduh", "Cek angka CO, kualitas data dan skor fuzzy; ekspor hasil.")])
    sample = sample_input(directory, key)
    manual, batch, training = st.tabs(["Input manual", "Unggah CSV", "Pelatihan ulang"])
    with manual:
        st.markdown("**Riwayat sensor (contoh historis dapat diedit)**")
        st.caption("Ganti angka dengan hasil pengukuran Anda. Gunakan timestamp jam penuh tanpa zona waktu, "
                   "dan isi –200 atau kosong apabila pembacaan hilang. Klik dua kali pada sel untuk mengedit.")
        editor_key = f"manual_{key}_{bundle_hash[:12]}"
        edited = st.data_editor(sample, hide_index=True, width="stretch", num_rows="fixed", key=editor_key,
                                column_config={"timestamp": st.column_config.TextColumn("Timestamp input (lokal)", width="medium"),
                                               **{col: st.column_config.NumberColumn(col, format="%.3f") for col in cols}})
        signature = sha256(edited.to_csv(index=False).encode()).hexdigest()
        state_key = f"prediction_{key}_{bundle_hash[:12]}"
        if st.button("Hitung prakiraan", type="primary"):
            try:
                row = forecast(edited, directory, key).iloc[-1]
                st.session_state[state_key] = (signature, row)
            except (ValueError, TypeError, UnicodeError, OSError) as error:
                st.session_state.pop(state_key, None)
                st.error(str(error))
        result = st.session_state.get(state_key)
        if result and result[0] == signature:
            row = result[1]
            if pd.isna(row.CO_mg_m3):
                st.warning(row.tindak_lanjut)
            else:
                st.subheader(f"Prakiraan pada {row.timestamp_target}")
                metric_row([("CO (mg/m³)", f"{row.CO_mg_m3:.3f}"), ("Skor pemantauan", f"{row.skor:.2f} / 100"),
                            ("Kelengkapan input", f"{row.kelengkapan_pct:.0f}%")])
                st.success(f"{row.prioritas} · {row.tindak_lanjut}")
                note("Referensi galat historis", f"{row.referensi_bawah:.3f} – {row.referensi_atas:.3f} mg/m³ "
                     "(prakiraan ± persentil ke-90 galat absolut validasi). Ini bukan interval kepercayaan terkalibrasi.")
                if row.kualitas_input != "Lengkap":
                    st.warning(f"{row.kualitas_input}. {row.jumlah_imputasi} imputasi; "
                               f"{row.di_luar_rentang_latih} pembacaan di luar rentang latih. Verifikasi data masukan.")
                if signature == sha256(sample.to_csv(index=False).encode()).hexdigest():
                    truth = predictions.loc[predictions.timestamp_target == row.timestamp_target, "aktual"]
                    if len(truth):
                        st.caption(f"Contoh historis: CO aktual {truth.iloc[0]:.3f} mg/m³; nilai aktual tidak diberikan ke model.")
                st.download_button("Unduh hasil manual", row.to_frame().T.to_csv(index=False),
                                   "prakiraan_manual.csv", "text/csv")
        elif result:
            st.caption("Input telah berubah. Jalankan Hitung prakiraan lagi.")
        with st.expander("Rentang sensor pada data latih"):
            st.table(pd.DataFrame([{"Kanal": col, **info["train_ranges"][col]} for col in cols]).round(4))
    with batch:
        st.caption(f"Kolom wajib: timestamp, {', '.join(cols)}. CSV boleh memuat lebih banyak kolom numerik.")
        st.download_button("Unduh contoh CSV", sample.to_csv(index=False), "contoh_sensor.csv", "text/csv")
        csv = st.file_uploader("Riwayat per jam", type="csv", key="forecast_csv")
        if csv is not None:
            try:
                out = forecast(BytesIO(csv.getvalue()), directory, key)
                good = out.CO_mg_m3.notna()
                metric_row([("Target prakiraan", len(out)), ("Berhasil dihitung", int(good.sum())),
                            ("Data kurang", int((~good).sum()))])
                if len(out) > 1:
                    st.line_chart(out.set_index("timestamp_target")["CO_mg_m3"], color=COLORS[0])
                st.dataframe(out.round(4), hide_index=True, width="stretch")
                st.download_button("Unduh hasil prediksi", out.to_csv(index=False), "prakiraan_CO.csv", "text/csv")
            except (ValueError, TypeError, UnicodeError, OSError) as error:
                st.error(str(error))
    with training:
        st.write("Pelatihan ulang opsional melatih empat konfigurasi. Default memakai UCI; "
                 "untuk data lain, sertakan kedelapan fitur dan label CO(GT). Model aktif hanya untuk sesi ini.")
        st.caption("Untuk deployment yang stabil, latih offline dengan train.py lalu sertakan artefaknya.")
        training_csv = st.file_uploader("Data latih alternatif (opsional)", type="csv", key="train_csv")
        if st.button("Latih empat model"):
            try:
                with st.spinner("Menjalankan GA, PSO, dan empat pelatihan RNN..."):
                    raw = training_csv.getvalue() if training_csv else official_csv(Path(".cache") / "AirQualityUCI.csv")
                    data = read_data(BytesIO(raw), [*FEATURES, TARGET])
                    if len(data) > 50_000:
                        raise ValueError("Maksimum 50.000 jam untuk pelatihan melalui dashboard; gunakan train.py untuk data besar.")
                    temp = TemporaryDirectory(prefix="co-run-")
                    active = run(data, Path(temp.name) / "session", sha256(raw).hexdigest())
                    verify_bundle(active)
                    st.session_state["run_temp"] = temp
                    st.session_state["active_run"] = str(active)
                st.rerun()
            except (OSError, ValueError, TypeError, UnicodeError, FloatingPointError) as error:
                st.error(str(error))


else:
    page_title(7, "Perbandingan model", "Ablation study: empat model dilatih independen pada protokol yang sama")
    metric_row([("Target uji identik", f"{len(predictions):,}"),
                ("MAE terbaik", f"{best_mae['MAE uji']:.4f}", best_mae["Model"]),
                ("RMSE terbaik", f"{best_rmse['RMSE uji']:.4f}", best_rmse["Model"]),
                ("R² terbaik", measure(best_r2["R² uji"], 4) if best_r2 is not None else "—",
                 best_r2["Model"] if best_r2 is not None else "Target konstan")])
    metric_columns = ["Model", "Fitur", "MAE uji", "RMSE uji", "R² uji", "RMSE validasi", "Bias uji"]
    st.dataframe(model_table[metric_columns].round(4), hide_index=True, width="stretch")
    with st.container(border=True):
        st.subheader("Perbandingan galat empat konfigurasi")
        short = ["Dasar", "GA", "PSO", "GA+PSO"]
        long = model_table.assign(Konfigurasi=short).melt(["Model", "Konfigurasi"], ["MAE uji", "RMSE uji"],
                                                          var_name="Metrik", value_name="Galat")
        encoded = alt.Chart(long).encode(
            x=alt.X("Konfigurasi:N", sort=short, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Metrik:N", sort=["MAE uji", "RMSE uji"]),
            y=alt.Y("Galat:Q", title="Galat (mg/m³)", scale=alt.Scale(domain=[0, 1.20 * max(long.Galat.max(), .01)])),
            color=alt.Color("Metrik:N", title=None, scale=alt.Scale(domain=["MAE uji", "RMSE uji"], range=COLORS[:2])),
            tooltip=["Model:N", "Metrik:N", alt.Tooltip("Galat:Q", format=".4f")])
        show_altair(encoded.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5) +
                    encoded.mark_text(dy=-8, color=INK, fontSize=11).encode(text=alt.Text("Galat:Q", format=".3f")), 260)
    note("Kesimpulan terukur", f"MAE terendah: {best_mae['Model']}. RMSE terendah: {best_rmse['Model']}. "
         "Peringkat tidak harus sama pada setiap metrik; fuzzy tidak mengubah galat RNN.", "good")
    with st.expander("Cara membaca metrik evaluasi"):
        st.markdown("**MAE** adalah rata-rata nilai absolut selisih prakiraan dan aktual (lebih kecil lebih baik).")
        st.markdown("**RMSE** adalah akar rata-rata kuadrat selisih; galat besar diberikan penalti lebih tinggi (lebih kecil lebih baik).")
        st.markdown("**R²** adalah koefisien determinasi, menunjukkan proporsi variasi target yang dijelaskan (lebih tinggi lebih baik).")
    perf = info["persistence"]
    st.caption(f"Pembanding persistensi (CO referensi terakhir): MAE {perf['mae']:.3f}, "
               f"RMSE {perf['rmse']:.3f} mg/m³ dan R² {measure(perf['r2'])}.")
    with st.expander("Parameter & biaya tiap konfigurasi"):
        st.dataframe(model_table[["Model", "Fitur", "Jendela (jam)", "Unit", "LR", "Dropout", "Parameter",
                                  "Pencarian (s)", "Latih akhir (s)"]].round(5), hide_index=True, width="stretch")
        st.caption("Waktu pencarian mengikuti pencatatan eksperimen bersama; cache kandidat dapat dipakai lintas seed. "
                   "Waktu pelatihan akhir dipisahkan dari waktu GA/PSO.")
        st.download_button("Unduh tabel pembanding lengkap", model_table.to_csv(index=False), "ablation.csv", "text/csv")
    st.subheader("Analisis galat per model")
    key = st.selectbox("Kurva prediksi", list(MODELS), format_func=MODELS.get, index=3)
    hours = st.slider("Rentang grafik (jam)", 24, 720, 168, 24)
    span = int((predictions.timestamp_target.max() - predictions.timestamp_target.min()).total_seconds() // 3600) + 1
    if span > hours:
        offset = st.slider("Mulai dari jam ke-", 0, span-hours, 0)
    else:
        offset = 0
    actual_plot(predictions, key, hours, offset)
    st.dataframe(error_groups(predictions, key).round(4), hide_index=True, width="stretch")
    spec = info["models"][key]
    st.caption(f"Kelompok CO di atas hanya untuk analisis galat, bukan kategori risiko. Persentil 90 galat absolut "
               f"validasi: {spec['val_abs_error_p90']:.3f} mg/m³. Pada uji, "
               f"{100 * spec['test_within_val_p90']:.1f}% target memiliki galat absolut ≤ nilai tersebut.")
    worst = predictions.assign(galat=predictions[key] - predictions.aktual)
    worst = worst.loc[worst.galat.abs().nlargest(5).index, ["timestamp_target", "aktual", key, "galat"]]
    st.markdown("**Lima galat absolut terbesar**")
    st.dataframe(worst.round(3), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.download_button("Unduh prediksi uji", (directory / "test_predictions.csv").read_bytes(),
                      "prediksi_uji.csv", "text/csv")
    b.download_button("Unduh prediksi validasi", (directory / "validation_predictions.csv").read_bytes(),
                      "prediksi_validasi.csv", "text/csv")
    st.caption("Batasan: satu dataset historis, satu split temporal, dan satu inisialisasi untuk pelatihan akhir. "
               "Hasil tidak menjamin kinerja pada kota, tahun, atau sensor yang berbeda.")

footer()
