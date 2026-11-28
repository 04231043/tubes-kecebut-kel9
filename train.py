"""Train and seal a self-contained evidence bundle for a one-hour CO forecast."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from air_data import FEATURES, MIN_STEPS, MAX_STEPS, TARGET, UCI_CSV_SHA256, make_windows, official_csv, prepare, read_data
from inference import estimate
from optimize import PENALTY, SEEDS, POP, GENERATIONS, PARTICLES, ROUNDS, genetic, swarm
from recurrent import BASE, fit

MODELS = {"baseline": "RNN dasar", "ga": "GA + RNN", "pso": "PSO + RNN", "hybrid": "GA + PSO + RNN"}
SEARCH_N, SEARCH_EPOCHS, FINAL_EPOCHS, PATIENCE = 1024, 5, 32, 7


def metrics(y, p):
    residual = p.astype(float) - y
    total = np.sum((y - y.mean()) ** 2)
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": float(1 - (residual @ residual) / total) if total > 0 else None,
            "bias": float(residual.mean())}


def data_summary(frame, anchors, cuts, folder):
    result = []
    for col, series in frame.items():
        values = series.dropna()
        hist, edges = np.histogram(values, bins=18) if len(values) else ([], [])
        result.append({"variabel": col, "missing": int(series.isna().sum()), "observed": int(len(values)),
                       "min": float(values.min()) if len(values) else None,
                       "mean": float(values.mean()) if len(values) else None,
                       "std": float(values.std(ddof=0)) if len(values) else None,
                       "median": float(values.median()) if len(values) else None,
                       "max": float(values.max()) if len(values) else None,
                       "hist": np.asarray(hist).tolist(), "edges": np.asarray(edges).tolist()})
    frame.resample("D").mean().to_csv(folder / "daily.csv", float_format="%.5f")
    frame[[c for c in [TARGET, *FEATURES] if c in frame]].corr(min_periods=2).to_csv(
        folder / "correlation.csv", float_format="%.6f")
    splits = {}
    boundaries = (0, *cuts, len(frame))
    for n, (key, ids) in enumerate(anchors.items()):
        left, right = boundaries[n:n+2]
        splits[key] = {"from": str(frame.index[left]), "to": str(frame.index[right-1]), "hours": right-left,
                       "observed_target": int(frame[TARGET].iloc[left:right].count()), "n": len(ids)}
    return {"rows": len(frame), "raw_columns": len(frame.columns) + 2,
            "summary": result, "splits": splits, "from": str(frame.index[0]), "to": str(frame.index[-1])}


def run(frame, out, digest, seeds=SEEDS):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Gunakan direktori keluaran baru. {out.name} sudah ada.")
    prep, x, y, parts, cuts = prepare(frame)
    yn = ((y - prep["y_mean"]) / prep["y_std"]).astype("float32")
    tr, va, te = (parts[k] for k in ["train", "val", "test"])
    all_cols = tuple(range(len(FEATURES)))
    pilot = tr[np.linspace(0, len(tr)-1, min(len(tr), SEARCH_N), dtype=int)]
    windows = {l: make_windows(x, l) for l in range(MIN_STEPS, MAX_STEPS+1)}

    def take(ids, ix, length):
        return np.ascontiguousarray(windows[length][ids-length][:, :, list(ix)])

    @lru_cache(maxsize=None)
    def cached(ix, config):
        cfg = dict(config)
        xt, xv = (take(k, ix, cfg["length"]) for k in (pilot, va))
        weights, _, _ = fit(xt, yn[pilot], xv, yn[va], cfg, 271, SEARCH_EPOCHS, SEARCH_EPOCHS)
        return metrics(y[va], estimate(xv, weights, prep))["rmse"]

    def objective(ix, params):
        return cached(tuple(ix), tuple(sorted(params.items())))

    def timed_search(name, procedure):
        records = []
        for seed in seeds:
            start = perf_counter()
            hits = cached.cache_info().misses
            record = procedure(seed)
            record["seconds"] = perf_counter() - start
            record["new_fits"] = cached.cache_info().misses - hits
            records.append(record)
            print(name, seed, round(record["fitness"], 5), flush=True)
        return records

    ga = timed_search("GA", lambda s: genetic(objective, s))
    features = min(ga, key=lambda r: r["fitness"])["features"]
    selected = tuple(FEATURES.index(c) for c in features)
    pso_full = timed_search("PSO 8", lambda s: swarm(objective, all_cols, s))
    pso_ga = timed_search("PSO GA", lambda s: swarm(objective, selected, s))
    get_params = lambda log: min(log, key=lambda r: r["fitness"])["parameters"]
    specs = {"baseline": (all_cols, BASE), "ga": (selected, BASE),
             "pso": (all_cols, get_params(pso_full)), "hybrid": (selected, get_params(pso_ga))}
    with TemporaryDirectory(dir=out.parent) as temp:
        folder = Path(temp) / "bundle"
        folder.mkdir()
        observed_train = frame[FEATURES].iloc[:cuts[0]]
        train_ranges = {
            col: {"min": float(observed_train[col].min()), "max": float(observed_train[col].max())}
            for col in FEATURES
        }
        result = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source_kind": "UCI Air Quality" if digest == UCI_CSV_SHA256 else "CSV pengguna",
                  "source_sha256": digest, "preprocessing": prep, "data": data_summary(frame, parts, cuts, folder),
                  "train_ranges": train_ranges,
                  "seeds": list(seeds), "search_train": len(pilot), "search_epochs": SEARCH_EPOCHS,
                  "search_init_seed": 271, "final_init_seed": 42, "max_epochs": FINAL_EPOCHS,
                  "patience": PATIENCE, "ga_penalty": PENALTY,
                  "search_budget": {"ga_population": POP, "ga_generations": GENERATIONS,
                                    "pso_particles": PARTICLES, "pso_iterations": ROUNDS},
                  "ga": ga, "pso_full": pso_full, "pso_ga": pso_ga,
                  "models": {}}
        validation = pd.DataFrame({"timestamp_target": frame.index[va], "aktual": y[va]})
        # Freeze every model and checkpoint with train/validation before scoring any test predictions.
        for key, (ix, cfg) in specs.items():
            start = perf_counter()
            xt, xv = (take(ids, ix, cfg["length"]) for ids in (tr, va))
            weight, history, best_epoch = fit(xt, yn[tr], xv, yn[va], cfg, 42, FINAL_EPOCHS, PATIENCE)
            np.savez_compressed(folder / (key + ".npz"), **weight)
            vp = estimate(xv, weight, prep)
            validation[key] = vp
            result["models"][key] = {"label": MODELS[key], "features": [FEATURES[i] for i in ix],
                "params": cfg, "val": metrics(y[va], vp), "best_epoch": best_epoch,
                "n_weights": sum(int(v.size) for v in weight.values()),
                "val_abs_error_p90": float(np.quantile(np.abs(vp - y[va]), .90)),
                "history": history, "fit_seconds": perf_counter()-start}
        validation.to_csv(folder / "validation_predictions.csv", index=False)
        predictions = pd.DataFrame({"timestamp_target": frame.index[te], "aktual": y[te]})
        for key, (ix, cfg) in specs.items():
            with np.load(folder / (key + ".npz"), allow_pickle=False) as checkpoint:
                preds = estimate(take(te, ix, cfg["length"]), checkpoint, prep)
            predictions[key] = preds
            result["models"][key]["test"] = metrics(y[te], preds)
            band = result["models"][key]["val_abs_error_p90"]
            result["models"][key]["test_within_val_p90"] = float(np.mean(np.abs(preds - y[te]) <= band))
        previous = frame[TARGET].shift(1).ffill().fillna(frame[TARGET].iloc[:cuts[0]].median())
        predictions["persistensi"] = previous.iloc[te].to_numpy()
        result["persistence"] = metrics(y[te], predictions["persistensi"].to_numpy())
        predictions.to_csv(folder / "test_predictions.csv", index=False)
        length = specs["hybrid"][1]["length"]
        demo_id = next((i for i in te if i >= 24 and frame[features].iloc[i-24:i].notna().all().all()), None)
        if demo_id is None:
            demo_id = int(te[0])
        demo = frame[features].iloc[demo_id-length:demo_id]
        demo.to_csv(folder / "contoh_sensor.csv")
        frame[FEATURES].iloc[demo_id-MAX_STEPS:demo_id].to_csv(folder / "contoh_sensor_lengkap.csv")
        result["demo_target"] = str(frame.index[demo_id])
        (folder / "experiment.json").write_text(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        manifest = {"schema_version": 1, "files": {
            f.name: sha256(f.read_bytes()).hexdigest() for f in sorted(folder.iterdir()) if f.is_file()}}
        (folder / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        folder.rename(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV UCI atau file data latih yang kompatibel")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    raw = args.csv.read_bytes() if args.csv else official_csv(".cache/AirQualityUCI.csv")
    frame = read_data(BytesIO(raw), [*FEATURES, TARGET])
    destination = args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    print(run(frame, destination, sha256(raw).hexdigest()))


if __name__ == "__main__":
    main()
