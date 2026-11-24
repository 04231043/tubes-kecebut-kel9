"""Hourly UCI observations, chronological partitions and train-only transforms."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_CSV_SHA256 = "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996"
TARGET = "CO(GT)"
FEATURES = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
            "PT08.S5(O3)", "T", "RH", "AH"]
MIN_STEPS, MAX_STEPS = 4, 8


def official_csv(cache):
    path = Path(cache)
    if path.exists():
        csv = path.read_bytes()
    else:
        with urlopen(UCI_URL, timeout=60) as response, ZipFile(BytesIO(response.read())) as archive:
            csv = archive.read("AirQualityUCI.csv")
        if sha256(csv).hexdigest() == UCI_CSV_SHA256:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(csv)
    if sha256(csv).hexdigest() != UCI_CSV_SHA256:
        raise ValueError("Berkas cache berbeda dari CSV UCI untuk eksperimen ini.")
    return csv


def numbers(series):
    text = series.astype("string").str.strip().replace("", pd.NA).str.replace(",", ".", regex=False)
    values = pd.to_numeric(text, errors="coerce").astype(float)
    if (text.notna() & values.isna()).any() or np.isinf(values).any():
        raise ValueError(f"Kolom {series.name} berisi angka yang tidak valid.")
    return values.replace(-200, np.nan)


def read_data(file, required, require_contiguous=True):
    """Accept the original UCI CSV or an ISO timestamp CSV, including forecast-only inputs."""
    data = file if isinstance(file, pd.DataFrame) else pd.read_csv(
        file, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    data = data.dropna(how="all").copy()
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.str.startswith("Unnamed:")]
    if data.empty:
        raise ValueError("CSV tidak memuat baris observasi.")
    if data.columns.duplicated().any():
        raise ValueError("Nama kolom harus unik.")
    if {"Date", "Time"} <= set(data.columns):
        dates = data.pop("Date").astype(str) + " " + data.pop("Time").astype(str)
        times = pd.to_datetime(dates, format="%d/%m/%Y %H.%M.%S", errors="coerce")
    else:
        key = next((k for k in ["timestamp", "Date_Time"] if k in data), None)
        if key is None:
            raise ValueError("Sertakan timestamp ISO, atau pasangan Date dan Time seperti pada CSV UCI.")
        times = pd.to_datetime(data.pop(key), format="ISO8601", errors="coerce")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Ada waktu kosong, tidak valid, atau duplikat.")
    if not times.dt.floor("h").equals(times):
        raise ValueError("Setiap waktu observasi harus tepat pada jam penuh.")
    absent = [name for name in required if name not in data]
    if absent:
        raise ValueError("Kolom wajib belum ada: " + ", ".join(absent))
    values = data.apply(numbers)
    values.index = pd.DatetimeIndex(times, name="timestamp")
    values = values.sort_index()
    if require_contiguous and not values.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Riwayat harus berurutan per jam. Isi jam yang hilang sebagai baris kosong beserta timestamp.")
    return values


def fit_transform(frame, end_train):
    training = frame.iloc[:end_train]
    x = training[FEATURES]
    med = x.median()
    if med.isna().any() or training[TARGET].dropna().empty:
        raise ValueError("Data latih tidak cukup untuk menghitung parameter.")
    filled = x.fillna(med)
    y = training[TARGET].dropna()
    return {"features": FEATURES, "median": med.tolist(), "mean": filled.mean().tolist(),
            "std": filled.std(ddof=0).replace(0, 1).tolist(), "y_mean": float(y.mean()),
            "y_std": float(y.std(ddof=0)) or 1.0}


def standardize(frame, columns, prep):
    ids = [prep["features"].index(c) for c in columns]
    raw = frame[columns].to_numpy(dtype=float)
    filled = np.where(np.isnan(raw), np.asarray(prep["median"])[ids], raw)
    return ((filled - np.asarray(prep["mean"])[ids]) / np.asarray(prep["std"])[ids]).astype("float32")


def make_windows(arr, length):
    return np.lib.stride_tricks.sliding_window_view(arr, length, axis=0).swapaxes(1, 2)


def prepare(frame):
    n = len(frame)
    a, b = int(n * .70), int(n * .85)
    prep = fit_transform(frame, a)
    x = standardize(frame, FEATURES, prep)
    target = frame[TARGET].to_numpy()
    # A target at i only sees inputs through i-1. Every model uses the same target times.
    anchors = np.arange(MAX_STEPS, n)
    present = make_windows(frame[FEATURES].notna().to_numpy(), MIN_STEPS)
    usable = present[anchors - MIN_STEPS].any(axis=1).all(axis=1)
    anchors = anchors[usable & np.isfinite(target[anchors])]
    splits = {"train": anchors[anchors < a], "val": anchors[(anchors >= a) & (anchors < b)],
              "test": anchors[anchors >= b]}
    if any(len(part) < 2 for part in splits.values()):
        raise ValueError("Setiap partisi memerlukan sedikitnya dua target CO dengan riwayat sensor.")
    return prep, x, target, splits, (a, b)
