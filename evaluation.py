"""Pure reporting helpers shared between the dashboard and validation tests."""
import numpy as np
import pandas as pd

MODEL_LABELS = {"baseline": "RNN dasar", "ga": "GA + RNN", "pso": "PSO + RNN", "hybrid": "GA + PSO + RNN"}


def search_seconds(info, model):
    phases = {"baseline": [], "ga": ["ga"], "pso": ["pso_full"], "hybrid": ["ga", "pso_ga"]}[model]
    return sum(log["seconds"] for phase in phases for log in info[phase])


def comparison(info):
    rows = []
    for key, label in MODEL_LABELS.items():
        spec = info["models"][key]
        p, t = spec["params"], spec["test"]
        rows.append({"Model": label, "Fitur": len(spec["features"]), "Jendela (jam)": p["length"],
                     "Unit": p["units"], "LR": p["lr"], "Dropout": p["dropout"],
                     "Parameter": spec["n_weights"], "MAE uji": t["mae"], "RMSE uji": t["rmse"],
                     "R² uji": t["r2"], "Bias uji": t["bias"], "RMSE validasi": spec["val"]["rmse"],
                     "Pencarian (s)": search_seconds(info, key), "Latih akhir (s)": spec["fit_seconds"]})
    return pd.DataFrame(rows)


def error_groups(predictions, key):
    actual = predictions["aktual"].to_numpy()
    residual = predictions[key].to_numpy() - actual
    masks = [actual <= 2, (actual > 2) & (actual <= 4), actual > 4]
    rows = []
    for name, mask in zip(["CO ≤ 2", "2 < CO ≤ 4", "CO > 4"], masks):
        r = residual[mask]
        rows.append({"Rentang CO aktual (mg/m³)": name, "n": int(mask.sum()),
                     "MAE (mg/m³)": float(np.abs(r).mean()) if len(r) else np.nan,
                     "RMSE (mg/m³)": float(np.sqrt((r ** 2).mean())) if len(r) else np.nan,
                     "Bias (mg/m³)": float(r.mean()) if len(r) else np.nan})
    return pd.DataFrame(rows)
