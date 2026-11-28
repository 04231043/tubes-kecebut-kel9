"""Eight-page Streamlit dashboard for the TE2514031 air-quality project."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from air_data import FEATURES, TARGET, official_csv, read_data
from dashboard_charts import (
    convergence, correlation_matrix, forecast_scatter, fuzzy_input, fuzzy_output,
    grouped_bars, histogram, missing_values, source_series, training_curve,
)
from dashboard_ui import (
    COLORS, footer, hero, init_style, overview_metrics,
    model_architecture, overview_results, page_header, sidebar_brand, sidebar_status, show_altair,
)
from evaluation import MODEL_LABELS as MODELS, comparison, error_groups
from fuzzy_system import ACTIONS, LEVELS, fuzzy_detail
from inference import DEFAULT, forecast, load_run, sample_input, verify_bundle
from train import run

st.set_page_config(page_title="Pantau CO · Air Quality Lab", page_icon=":material/air:", layout="wide")
PAGES = ["Ringkasan", "Eksplorasi Data", "Seleksi Fitur", "Optimasi PSO", "Jaringan Saraf",
         "Fuzzy Logic", "Prakiraan CO", "Evaluasi Model"]
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


def metric_row(items, key="page"):
    with st.container(key=f"metric_cards_{key}"):
        for col, item in zip(st.columns(len(items)), items):
            label, value, *rest = item
            col.metric(label, value, border=True, help=rest[0] if rest else None)


def page_title(index, title, description):
    page_header(PAGES[index], title, description)


def open_page(page):
    st.session_state["navigation"] = page


def actual_plot(predictions, column, hours, offset=0, height=255, forecast_label=None):
    subset = predictions.set_index("timestamp_target").sort_index()
    first = subset.index[0] + pd.Timedelta(hours=offset)
    timeline = pd.date_range(first, periods=hours, freq="h", name="Waktu target")
    forecast_label = forecast_label or MODELS[column]
    wide = (subset.reindex(timeline)[["aktual", column]]
            .rename(columns={"aktual": "Aktual", column: forecast_label}).reset_index())
    # Keep the source's wall-clock timestamps identical across browser time zones.
    wide["Waktu target"] = wide["Waktu target"].dt.tz_localize("UTC")
    long = wide.melt(id_vars="Waktu target", var_name="Seri", value_name="CO")
    time_scale = alt.Scale(type="utc")
    axis_format = "%d/%m" if hours >= 96 else "%d/%m %H:%M"
    time = alt.X("Waktu target:T", scale=time_scale,
                 axis=alt.Axis(title=None, format=axis_format, tickCount=5, labelOverlap="greedy",
                               labelFlush=True, labelBound=True, grid=False))
    co = alt.Y("CO:Q", title="CO (mg/m³)", scale=alt.Scale(zero=True))
    labels = ["Aktual", forecast_label]
    legend = alt.Legend(title=None, symbolType="stroke", symbolSize=240)
    series = alt.Color("Seri:N", legend=legend,
                       scale=alt.Scale(domain=labels, range=["#273d59", COLORS[0]]))
    stroke = alt.StrokeDash("Seri:N", legend=legend,
                            scale=alt.Scale(domain=labels, range=[[1, 0], [5, 3]]))
    lines = (alt.Chart(long).mark_line(strokeWidth=2.2, invalid="break-paths-filter-domains")
             .encode(x=time, y=co, color=series, strokeDash=stroke))
    nearest = alt.selection_point(fields=["Waktu target"], nearest=True, on="pointerover",
                                  clear="pointerout", empty=False)
    tooltip = [alt.Tooltip("Waktu target:T", timeUnit="utcyearmonthdatehoursminutes",
                           title="Waktu target", format="%d/%m/%Y %H:%M")]
    tooltip.extend(alt.Tooltip(f"{label}:Q", format=".3f") for label in ["Aktual", forecast_label])
    hover = (alt.Chart(wide).mark_rule(color="#8291a7", aria=False)
             .encode(x=alt.X("Waktu target:T", scale=time_scale), opacity=alt.when(nearest).then(alt.value(1)).otherwise(alt.value(0)),
                     tooltip=tooltip).add_params(nearest))
    points = alt.Chart(long).mark_circle(size=48, aria=False).encode(x=time, y=co, color=series).transform_filter(nearest)
    show_altair(lines + hover + points, height)


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
    verify_bundle(directory)
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
page = st.sidebar.radio("Navigasi", PAGES, key="navigation", label_visibility="collapsed", width="stretch")
sidebar_status(directory != DEFAULT)
st.sidebar.link_button("Sumber data UCI", "https://doi.org/10.24432/C59K5F", icon=":material/open_in_new:", width="stretch")
if directory != DEFAULT:
    st.sidebar.warning("Eksperimen sesi ini aktif.")
    if st.sidebar.button("Kembali ke eksperimen UCI"):
        st.session_state.pop("active_run", None)
        st.session_state.pop("run_temp", None)
        st.rerun()


if page == PAGES[0]:
    hero()
    with st.container(key="overview_actions", horizontal=True, gap="small"):
        st.button("Coba prakiraan", icon=":material/arrow_forward:", type="primary",
                  on_click=open_page, args=(PAGES[6],))
        st.button("Bandingkan model", icon=":material/bar_chart:", on_click=open_page, args=(PAGES[7],))
    left, right = st.columns([2.65, 1], gap="small", vertical_alignment="top")
    with left:
        with st.container(border=True, key="overview_chart"):
            st.subheader("Aktual vs prakiraan hibrida")
            st.caption("96 jam pertama uji · jeda menunjukkan jam tanpa target valid")
            actual_plot(predictions, "hybrid", 96, height=235, forecast_label="Prakiraan hibrida")
    with right:
        with st.container(border=True, key="overview_summary"):
            st.subheader("Hasil uji model")
            results = [("MAE terendah", best_mae, "MAE uji", "mg/m³"),
                       ("RMSE terendah", best_rmse, "RMSE uji", "mg/m³")]
            if best_r2 is not None:
                results.append(("R² tertinggi", best_r2, "R² uji", ""))
            overview_results([
                (label, f"{result[metric]:.4f}".replace(".", ","), unit, result["Model"])
                for label, result, metric, unit in results
            ])
    overview_metrics([
        ("Observasi per jam", f"{stats['rows']:,}".replace(",", "."),
         "UCI Air Quality" if directory == DEFAULT else "Dataset sesi"),
        ("Kanal model hibrida", f"{len(hybrid['features'])} / {len(FEATURES)}", "Dipilih oleh GA"),
        ("Waktu target uji", f"{len(predictions):,}".replace(",", "."), "Sama untuk empat model"),
    ])
    with st.container(border=True, key="overview_architecture"):
        st.subheader("Arsitektur model")
        model_architecture()
    with st.expander("Metode dan tim proyek"):
        st.dataframe(pd.DataFrame(MEMBERS, columns=["Nama", "NIM"]), hide_index=True, width="stretch")
        fit_notes()
        st.code(f"SHA-256 CSV: {info['source_sha256']}\nBundle: {bundle_hash}", language=None)
        st.download_button("Unduh log eksperimen JSON", (directory / "experiment.json").read_bytes(),
                           "experiment.json", "application/json")


elif page == PAGES[1]:
    page_title(1, "Eksplorasi Data", "Observasi historis sebelum imputasi. Kode −200 menandai data hilang.")
    summary = pd.DataFrame(stats["summary"]).set_index("variabel")
    table = summary[["observed", "missing", "min", "median", "max"]].copy()
    table["hilang (%)"] = table.missing / stats["rows"] * 100
    metric_row([("Jam tercatat", f"{stats['rows']:,}"),
                ("CO teramati", f"{int(summary.loc[TARGET, 'observed']):,}"),
                ("CO hilang", f"{table.loc[TARGET, 'hilang (%)']:.1f}%"),
                ("Kandidat masukan", len(FEATURES))])
    st.caption(f"Periode {stats['from']} — {stats['to']} (timestamp lokal sumber).")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Distribusi data hilang")
        show_altair(missing_values(table), 278)
    with right:
        st.subheader("Korelasi observasi")
        show_altair(correlation_matrix(corr, TARGET), 278)
        st.caption("Pearson berdasarkan pasangan nilai yang tersedia; ditampilkan untuk EDA, bukan untuk memilih fitur atau tuning.")
    st.subheader("Statistik deskriptif")
    st.dataframe(table.rename(columns={"observed": "Teramati", "missing": "Hilang",
                                       "min": "Min", "median": "Median", "max": "Maks"}).round(2), width="stretch")
    variable = st.selectbox("Variabel untuk melihat tren", list(summary.index), index=list(summary.index).index(TARGET))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rata-rata harian")
        show_altair(source_series(daily.set_index("timestamp")[variable], variable, daily=True), 235)
    with c2:
        st.subheader("Sebaran nilai")
        item = summary.loc[variable]
        if item["hist"]:
            show_altair(histogram(item["hist"], item["edges"], variable, "Jumlah observasi"), 235)
        else:
            st.info("Tidak ada nilai teramati untuk variabel ini.")
    st.subheader("Split kronologis 70 / 15 / 15")
    split = pd.DataFrame(stats["splits"]).T
    periods = {"train": "Latih", "val": "Validasi", "test": "Uji"}
    split_plot = (split[["hours", "n"]].rename(index=periods, columns={"hours": "Jam periode", "n": "Target valid"})
                  .rename_axis("Bagian").reset_index().melt(id_vars="Bagian", var_name="Metrik", value_name="Nilai"))
    show_altair(grouped_bars(split_plot, "Bagian", ["Latih", "Validasi", "Uji"], "Jumlah jam"), 188)
    st.dataframe(split.rename(columns={"from": "Awal", "to": "Akhir", "hours": "Jam periode",
                                       "observed_target": "CO teramati", "n": "Target valid"}), width="stretch")
    st.info("Jendela memakai sensor hingga t, prediksi CO pada t+1. Median dan standardisasi dipelajari dari latih; "
            "target CO hilang tidak diimputasi. Perbandingan memakai target valid yang sama.")
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
                column = st.selectbox("Kolom CSV", df.columns)
                if df[column].notna().any():
                    show_altair(source_series(df[column], column), 245)
                else:
                    st.info("Tidak ada nilai teramati untuk kolom ini.")
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
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Konvergensi setiap seed")
        show_altair(convergence(logs, "Generasi", "Fitness (mg/m³)"), 255)
    with right:
        st.subheader("Kromosom terpilih")
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
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Konvergensi gbest")
        show_altair(convergence(logs, "Iterasi", "RMSE validasi terbaik (mg/m³)"), 255)
    with right:
        st.subheader("Ruang pencarian")
        st.table(pd.DataFrame([
            ("Jendela", "4, 6, 8 jam", str(params["length"])),
            ("Unit", "8, 12, 16, 20, 24", str(params["units"])),
            ("Learning rate", "0,001 – 0,01 (log)", str(params["lr"])),
            ("Dropout", "0 – 0,25", str(params["dropout"]))
        ], columns=["Parameter", "Batas", "Terpilih"]))
    st.dataframe(pd.DataFrame([{"Seed": g["seed"], **g["parameters"], "RMSE pencarian": g["fitness"],
                               "Evaluasi": len(g["candidates"]), "Waktu (s)": g["seconds"]} for g in logs]).round(5),
                 hide_index=True, width="stretch")
    st.caption(f"{info['search_budget']['pso_particles']} partikel × {info['search_budget']['pso_iterations']} iterasi/seed. "
               f"Rata-rata gbest akhir {mean}; simpangan baku sampel {sd}. "
               f"Model akhir {MODELS[key]} memiliki RMSE validasi {info['models'][key]['val']['rmse']:.4f} mg/m³.")
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
    st.latex(r"h_t=\tanh(W_x x_t+W_h h_{t-1}+b_h),\qquad \widehat{y}_{t+1}=W_y h_t+b_y")
    h = pd.DataFrame(model["history"])
    st.subheader("Kurva pelatihan dan validasi")
    show_altair(training_curve(h, model["best_epoch"]), 255)
    st.subheader("Aktual dan prediksi")
    actual_plot(predictions, key, 168)
    a, b = st.columns(2)
    residual = predictions[key] - predictions.aktual
    with a:
        st.subheader("Sebaran aktual dan prakiraan")
        show_altair(forecast_scatter(predictions, key), 245)
        st.caption("Garis putus-putus menandai prakiraan yang sama dengan nilai aktual.")
    with b:
        st.subheader("Sebaran galat prakiraan")
        counts, edges = np.histogram(residual, bins=26)
        show_altair(histogram(counts, edges, "Prakiraan − aktual (mg/m³)", "Jumlah target", reference=0), 245)
    st.caption("Kurva dan scatter memakai checkpoint; galat positif = prediksi terlalu tinggi. "
               f"Bias = {t['bias']:.4f} mg/m³.")
    fit_notes()
    st.download_button("Unduh kurva pelatihan", h.to_csv(index=False), f"history_{key}.csv", "text/csv")


elif page == PAGES[5]:
    page_title(5, "Fuzzy Inference System", "Mamdani min–max dan centroid: prioritas verifikasi sensor")
    st.warning("Skor 0–100 adalah indeks internal proyek untuk pemantauan sensor, bukan AQI/ISPU resmi, "
               "batas paparan kesehatan, atau rekomendasi keselamatan.")
    c = st.slider("Konsentrasi CO contoh (mg/m³)", 0., 15., 3., .1)
    score, idx, deg, agg = fuzzy_detail(c)
    metric_row([("CO masukan", f"{c:.1f} mg/m³"), ("Centroid", f"{score:.2f} / 100"), ("Prioritas", LEVELS[idx])])
    left, right = st.columns(2)
    with left:
        st.subheader("Keanggotaan CO masukan")
        show_altair(fuzzy_input(c), 245)
    with right:
        st.subheader("Agregasi dan centroid")
        show_altair(fuzzy_output(score, agg), 245)
        st.caption("Area berwarna menunjukkan agregasi aturan. Garis putus-putus menandai centroid.")
    st.latex(r"s^*=\frac{\int_0^{100}s\,\mu_{agg}(s)\,ds}{\int_0^{100}\mu_{agg}(s)\,ds}")
    st.dataframe([{"Aturan": f"R{i+1}", "Jika CO": label, "Maka prioritas": label,
                   "Aktivasi": round(deg[i], 3), "Tindak lanjut": ACTIONS[i]} for i, label in enumerate(LEVELS)],
                 hide_index=True, width="stretch")
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
    page_title(6, "Prakiraan CO", "Gunakan model terlatih untuk menghitung konsentrasi karbon monoksida satu jam ke depan.")
    key = st.selectbox("Konfigurasi model", list(MODELS), format_func=MODELS.get, index=3)
    spec = info["models"][key]
    length, cols = spec["params"]["length"], spec["features"]
    st.info(f"Model menggunakan {length} jam berturut-turut × {len(cols)} kanal. "
            "Keluaran diprakirakan satu jam setelah observasi terakhir.")
    sample = sample_input(directory, key)
    manual, batch, training = st.tabs(["Input manual", "Unggah CSV", "Pelatihan ulang"])
    with manual:
        st.markdown("**Riwayat sensor (contoh historis dapat diedit)**")
        st.caption("Ganti angka dengan hasil pengukuran Anda. Gunakan timestamp jam penuh tanpa zona waktu, "
                   "dan isi –200 atau kosong apabila pembacaan hilang.")
        editor_key = f"manual_{key}_{bundle_hash[:12]}"
        edited = st.data_editor(sample, hide_index=True, width="stretch", num_rows="fixed", key=editor_key)
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
                            ("Kelengkapan input", f"{row.kelengkapan_pct:.0f}%")], key="manual")
                st.success(f"{row.prioritas} · {row.tindak_lanjut}")
                st.caption(f"Referensi galat validasi: {row.referensi_bawah:.3f}–{row.referensi_atas:.3f} mg/m³ "
                           f"(prakiraan ± persentil ke-90 galat absolut validasi). Ini bukan interval kepercayaan terkalibrasi.")
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
        st.caption(f"Kolom wajib: timestamp, {', '.join(cols)}. Kolom tambahan boleh disertakan.")
        st.download_button("Unduh contoh CSV", sample.to_csv(index=False), "contoh_sensor.csv", "text/csv")
        csv = st.file_uploader("Riwayat per jam", type="csv", key="forecast_csv")
        if csv is not None:
            try:
                out = forecast(BytesIO(csv.getvalue()), directory, key)
                good = out.CO_mg_m3.notna()
                metric_row([("Target prakiraan", len(out)), ("Berhasil dihitung", int(good.sum())),
                            ("Data kurang", int((~good).sum()))], key="batch")
                if len(out) > 1 and good.any():
                    st.subheader("Prakiraan per jam")
                    show_altair(source_series(out.set_index("timestamp_target")["CO_mg_m3"], "CO (mg/m³)"), 245)
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
    page_title(7, "Evaluasi Model", "Bandingkan galat dari empat konfigurasi RNN pada target uji yang sama.")
    metric_row([("Target uji identik", f"{len(predictions):,}"),
                ("MAE terbaik", f"{best_mae['MAE uji']:.4f}", best_mae["Model"]),
                ("RMSE terbaik", f"{best_rmse['RMSE uji']:.4f}", best_rmse["Model"]),
                ("R² terbaik", measure(best_r2["R² uji"], 4) if best_r2 is not None else "—",
                 best_r2["Model"] if best_r2 is not None else "Target konstan")])
    metric_columns = ["Model", "Fitur", "MAE uji", "RMSE uji", "R² uji", "RMSE validasi", "Bias uji"]
    st.dataframe(model_table[metric_columns].round(4), hide_index=True, width="stretch")
    st.subheader("Galat uji per konfigurasi")
    short = ["Dasar", "GA", "PSO", "GA + PSO"]
    errors = model_table[["MAE uji", "RMSE uji"]].assign(Model=short).melt(
        id_vars="Model", var_name="Metrik", value_name="Nilai",
    )
    show_altair(grouped_bars(errors, "Model", short, "Galat CO (mg/m³)"), 225)
    st.info(f"MAE terendah: {best_mae['Model']}. RMSE terendah: {best_rmse['Model']}. "
            "Peringkat tidak harus sama pada setiap metrik; fuzzy tidak mengubah galat RNN.")
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
