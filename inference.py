"""Shared interface for the dashboard, notebook and command-line checks."""
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd

from air_data import MIN_STEPS, make_windows, read_data, standardize
from fuzzy_system import compute_fuzzy_risk
from recurrent import predict

DEFAULT = Path(__file__).resolve().parent / "artifacts" / "uci-v1"
MODEL_KEYS = ("baseline", "ga", "pso", "hybrid")
BUNDLE_FILES = {"experiment.json", "daily.csv", "correlation.csv",
                "test_predictions.csv", "validation_predictions.csv",
                "contoh_sensor.csv", "contoh_sensor_lengkap.csv",
                *(key + ".npz" for key in MODEL_KEYS)}


def load_run(directory=DEFAULT):
    directory = Path(directory)
    return json.loads((directory / "experiment.json").read_text(encoding="utf-8"))


def verify_bundle(directory=DEFAULT):
    """Check the closed set of expected artifacts before loading a run."""
    directory = Path(directory)
    manifest = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("files", {})
    if set(hashes) != BUNDLE_FILES:
        raise ValueError("Daftar artefak tidak lengkap atau versi bundle tidak sesuai.")
    for name in sorted(BUNDLE_FILES):
        if sha256((directory / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Artefak {name} berubah. Jalankan ulang train.py atau pulihkan bundle asli.")
    return hashes


def estimate(x, weights, prep):
    return np.maximum(0, predict(x, weights) * prep["y_std"] + prep["y_mean"])


def sample_input(directory=DEFAULT, model="hybrid"):
    """All models can demonstrate the same historical target timestamp."""
    directory = Path(directory)
    result = load_run(directory)
    if model not in MODEL_KEYS:
        raise ValueError("Model tidak tersedia.")
    spec = result["models"][model]
    return (pd.read_csv(directory / "contoh_sensor_lengkap.csv")
            .tail(spec["params"]["length"])[["timestamp", *spec["features"]]]
            .reset_index(drop=True))


def forecast(source, directory=DEFAULT, model="hybrid"):
    directory = Path(directory)
    result = load_run(directory)
    if model not in MODEL_KEYS:
        raise ValueError("Model tidak tersedia.")
    spec = result["models"][model]
    cols, length = spec["features"], spec["params"]["length"]
    frame = read_data(source, cols, required_only=True)
    if len(frame) < length:
        raise ValueError(f"Sediakan setidaknya {length} jam observasi.")
    inputs = make_windows(standardize(frame, cols, result["preprocessing"]), length)
    # Require per-channel evidence in the last four hours of each input.
    recent = make_windows(frame[cols].notna().to_numpy(), MIN_STEPS)[length - MIN_STEPS:]
    valid = recent.any(axis=1).all(axis=1)
    missing = make_windows(frame[cols].isna().to_numpy(), length).sum(axis=(1, 2))
    lower = np.asarray([result["train_ranges"][col]["min"] for col in cols])
    upper = np.asarray([result["train_ranges"][col]["max"] for col in cols])
    raw = frame[cols].to_numpy(dtype=float)
    ood = make_windows((raw < lower) | (raw > upper), length).sum(axis=(1, 2))
    dates = frame.index[length - 1:] + pd.Timedelta(hours=1)
    values = np.full(len(dates), np.nan)
    if valid.any():
        with np.load(directory / f"{model}.npz", allow_pickle=False) as weights:
            values[valid] = estimate(inputs[valid], weights, result["preprocessing"])
    rows = []
    margin = spec["val_abs_error_p90"]
    for date, co, absent, outside in zip(dates, values, missing, ood):
        completeness = (1 - absent / (length * len(cols))) * 100
        quality = "Periksa rentang" if outside else ("Ada imputasi" if absent else "Lengkap")
        if np.isfinite(co):
            score, level, _, action = compute_fuzzy_risk(co)
            rows.append((date, co, score, level, action, quality, completeness, int(absent), int(outside),
                         max(0., co - margin), co + margin))
        else:
            rows.append((date, np.nan, np.nan, "Data kurang", "Lengkapi riwayat sensor empat jam terakhir.",
                         "Data kurang", completeness, int(absent), int(outside), np.nan, np.nan))
    return pd.DataFrame(rows, columns=["timestamp_target", "CO_mg_m3", "skor", "prioritas", "tindak_lanjut",
                                      "kualitas_input", "kelengkapan_pct", "jumlah_imputasi",
                                      "di_luar_rentang_latih", "referensi_bawah", "referensi_atas"])
