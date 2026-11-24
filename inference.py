"""Shared interface for the dashboard, notebook and command-line checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from air_data import MIN_STEPS, make_windows, read_data, standardize
from fuzzy_system import compute_fuzzy_risk
from recurrent import predict

DEFAULT = Path(__file__).resolve().parent / "artifacts" / "uci-v1"


def load_run(directory=DEFAULT):
    directory = Path(directory)
    return json.loads((directory / "experiment.json").read_text(encoding="utf-8"))


def estimate(x, weights, prep):
    return np.maximum(0, predict(x, weights) * prep["y_std"] + prep["y_mean"])


def forecast(source, directory=DEFAULT):
    directory = Path(directory)
    result = load_run(directory)
    spec = result["models"]["hybrid"]
    cols, length = spec["features"], spec["params"]["length"]
    frame = read_data(source, cols)
    if len(frame) < length:
        raise ValueError(f"Sediakan setidaknya {length} jam observasi.")
    inputs = make_windows(standardize(frame, cols, result["preprocessing"]), length)
    # Require per-channel evidence in the last four hours of each input.
    recent = make_windows(frame[cols].notna().to_numpy(), MIN_STEPS)[length - MIN_STEPS:]
    valid = recent.any(axis=1).all(axis=1)
    dates = frame.index[length - 1:] + pd.Timedelta(hours=1)
    values = np.full(len(dates), np.nan)
    if valid.any():
        with np.load(directory / "hybrid.npz", allow_pickle=False) as weights:
            values[valid] = estimate(inputs[valid], weights, result["preprocessing"])
    rows = []
    for date, co in zip(dates, values):
        if np.isfinite(co):
            score, level, _, action = compute_fuzzy_risk(co)
            rows.append((date, co, score, level, action))
        else:
            rows.append((date, np.nan, np.nan, "Data kurang", "Lengkapi riwayat sensor empat jam terakhir."))
    return pd.DataFrame(rows, columns=["timestamp_target", "CO_mg_m3", "skor", "prioritas", "tindak_lanjut"])
