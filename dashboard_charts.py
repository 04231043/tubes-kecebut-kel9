"""Browser-rendered charts for model analysis and historical observations."""
import altair as alt
import numpy as np
import pandas as pd

from dashboard_ui import COLORS, INK
from fuzzy_system import GRID, IN_CENTERS, LEVELS, OUT_CENTERS, memberships


def missing_values(table):
    values = table.rename_axis("Variabel").reset_index()
    return alt.Chart(values).mark_bar(color=COLORS[0], cornerRadiusEnd=3).encode(
        x=alt.X("hilang (%):Q", title="Nilai hilang (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Variabel:N", title=None, sort="-x", axis=alt.Axis(labelLimit=115)),
        tooltip=[alt.Tooltip("Variabel:N"), alt.Tooltip("missing:Q", title="Jam hilang", format=",.0f"),
                 alt.Tooltip("hilang (%):Q", title="Hilang (%)", format=".1f")],
    )


def correlation_matrix(correlations, target):
    def short_name(name):
        return "CO" if name == target else name.split("(")[0].replace("PT08.", "")

    cells = (correlations.rename_axis(index="Variabel Y", columns="Variabel X")
             .reset_index().melt(id_vars="Variabel Y", var_name="Variabel X", value_name="Korelasi"))
    cells["Kanal X"] = cells["Variabel X"].map(short_name)
    cells["Kanal Y"] = cells["Variabel Y"].map(short_name)
    cells["Angka"] = cells["Korelasi"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "—")
    order = [short_name(name) for name in correlations.columns]
    base = alt.Chart(cells).encode(
        x=alt.X("Kanal X:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Kanal Y:N", title=None, sort=order),
        tooltip=["Variabel X:N", "Variabel Y:N", alt.Tooltip("Korelasi:Q", format=".3f")],
    )
    tiles = base.mark_rect(cornerRadius=2, stroke="white").encode(
        color=alt.Color("Korelasi:Q", title="Pearson r", legend=alt.Legend(direction="horizontal"),
                        scale=alt.Scale(domain=[-1, 0, 1], range=["#255492", "#f2f5f4", "#a4492d"])),
    )
    labels = base.mark_text(fontSize=10).encode(
        text="Angka:N", color=alt.condition("abs(datum.Korelasi) > 0.68", alt.value("white"), alt.value(INK)),
    )
    return tiles + labels


def source_series(values, ylabel, *, daily=False):
    # Encode timezone-free sensor clocks without browser timezone shifts.
    timestamps = values.index.tz_localize("UTC")
    series = pd.DataFrame({"Waktu": timestamps, "Nilai": values.to_numpy()})
    long_range = timestamps.max() - timestamps.min() >= pd.Timedelta(days=90)
    axis_format = "%b %Y" if long_range else "%d/%m" if daily else "%d/%m %H:%M"
    time_unit = "utcyearmonthdate" if daily else "utcyearmonthdatehoursminutes"
    time_format = "%d/%m/%Y" if daily else "%d/%m/%Y %H:%M"
    # Keep isolated valid values visible when there are few observations among gaps.
    dots = alt.OverlayMarkDef(size=30) if values.count() <= 100 else False
    return alt.Chart(series).mark_line(
        color=COLORS[0], strokeWidth=2, point=dots, invalid="break-paths-filter-domains",
    ).encode(
        x=alt.X("Waktu:T", title=None, scale=alt.Scale(type="utc"),
                axis=alt.Axis(format=axis_format, tickCount=5, labelOverlap="greedy")),
        y=alt.Y("Nilai:Q", title=ylabel, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Waktu:T", timeUnit=time_unit, format=time_format),
                 alt.Tooltip("Nilai:Q", title=ylabel, format=".3f")],
    )


def histogram(counts, edges, xlabel, ylabel, reference=None):
    values = pd.DataFrame({"Awal": edges[:-1], "Akhir": edges[1:], "Jumlah": counts})
    bars = alt.Chart(values).mark_bar(color=COLORS[1], opacity=.84, binSpacing=1).encode(
        x=alt.X("Awal:Q", title=xlabel, bin="binned"), x2="Akhir:Q",
        y=alt.Y("Jumlah:Q", title=ylabel, axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("Awal:Q", format=".3f"), alt.Tooltip("Akhir:Q", format=".3f"), "Jumlah:Q"],
    )
    return bars if reference is None else bars + marker(reference, "Galat nol")


def grouped_bars(values, category, order, xlabel):
    series = values["Metrik"].drop_duplicates().tolist()
    return alt.Chart(values).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Nilai:Q", title=xlabel),
        y=alt.Y(f"{category}:N", title=None, sort=order),
        yOffset=alt.YOffset("Metrik:N", sort=series),
        color=alt.Color("Metrik:N", legend=alt.Legend(title=None),
                        scale=alt.Scale(domain=series, range=COLORS[:len(series)])),
        tooltip=[f"{category}:N", "Metrik:N", alt.Tooltip("Nilai:Q", format=".4~f")],
    )


def metric_lines(values, xlabel, ylabel, series, colors, *, dots=False, domain=None):
    return alt.Chart(values).mark_line(
        strokeWidth=2, point=alt.OverlayMarkDef(size=42) if dots else False,
    ).encode(
        x=alt.X("Langkah:Q", title=xlabel, axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Nilai:Q", title=ylabel, scale=alt.Scale(zero=False, domain=domain) if domain else alt.Scale(zero=False)),
        color=alt.Color("Seri:N", legend=alt.Legend(title=None, columns=3),
                        scale=alt.Scale(domain=series, range=colors)),
        tooltip=["Seri:N", alt.Tooltip("Langkah:Q", title=xlabel), alt.Tooltip("Nilai:Q", title=ylabel, format=".3f")],
    )


def convergence(logs, xlabel, ylabel):
    values = pd.DataFrame([
        {"Langkah": step, "Nilai": value, "Seri": f"Seed {log['seed']}"}
        for log in logs for step, value in enumerate(log["history"], 1)
    ])
    series = values["Seri"].drop_duplicates().tolist()
    return metric_lines(values, xlabel, ylabel, series, COLORS[:len(series)], dots=True)


def marker(value, label):
    return alt.Chart(pd.DataFrame({"Tanda": [value], "Keterangan": [label]})).mark_rule(
        color=INK, strokeDash=[5, 4], strokeWidth=1.5,
    ).encode(x="Tanda:Q", tooltip=["Keterangan:N"])


def training_curve(history, best_epoch):
    values = (history.rename(columns={"epoch": "Langkah", "train_mse_z": "Latih", "val_mse_z": "Validasi"})
              .melt(id_vars="Langkah", value_vars=["Latih", "Validasi"], var_name="Seri", value_name="Nilai"))
    lines = metric_lines(values, "Epoch", "MSE pada target baku", ["Latih", "Validasi"], [COLORS[0], COLORS[2]])
    return lines + marker(best_epoch, f"Checkpoint pada epoch {best_epoch}")


def forecast_scatter(predictions, model):
    values = predictions[["timestamp_target", "aktual", model]].rename(columns={"aktual": "Aktual", model: "Prakiraan"})
    values["Waktu target"] = values.pop("timestamp_target").dt.strftime("%d/%m/%Y %H:%M")
    lower = min(0, values[["Aktual", "Prakiraan"]].min().min())
    upper = max(.01, values[["Aktual", "Prakiraan"]].max().max())
    scale = alt.Scale(domain=[float(lower), float(upper)])
    x = alt.X("Aktual:Q", title="CO aktual (mg/m³)", scale=scale)
    y = alt.Y("Prakiraan:Q", title="CO prakiraan (mg/m³)", scale=scale)
    points = alt.Chart(values).mark_point(filled=True, opacity=.35, size=25, color=COLORS[0]).encode(
        x=x, y=y, tooltip=["Waktu target:N", alt.Tooltip("Aktual:Q", format=".3f"),
                           alt.Tooltip("Prakiraan:Q", format=".3f")],
    )
    line = alt.Chart(pd.DataFrame({"Aktual": [lower, upper], "Prakiraan": [lower, upper]})).mark_line(
        color=COLORS[2], strokeDash=[5, 4], strokeWidth=1.5,
    ).encode(x=x, y=y)
    return line + points


def fuzzy_input(co):
    grid = np.linspace(0, 15, 181)
    return membership_curves(grid, IN_CENTERS, "CO (mg/m³)") + marker(co, f"CO masukan {co:.1f} mg/m³")


def fuzzy_output(score, aggregate):
    area = alt.Chart(pd.DataFrame({"Langkah": GRID, "Nilai": aggregate})).mark_area(
        color=COLORS[0], opacity=.25,
    ).encode(
        x="Langkah:Q", y="Nilai:Q",
        tooltip=[alt.Tooltip("Langkah:Q", title="Skor prioritas", format=".1f"),
                 alt.Tooltip("Nilai:Q", title="Agregasi maksimum", format=".3f")],
    )
    lines = membership_curves(GRID, OUT_CENTERS, "Skor prioritas")
    return lines + area + marker(score, f"Centroid {score:.2f}")


def membership_curves(grid, centers, xlabel):
    values = pd.DataFrame(memberships(grid, centers), columns=LEVELS).assign(Langkah=grid).melt(
        id_vars="Langkah", var_name="Seri", value_name="Nilai",
    )
    return metric_lines(values, xlabel, "Derajat keanggotaan", LEVELS, COLORS, domain=[0, 1.04])
