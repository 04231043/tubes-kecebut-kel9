"""Elman RNN with backpropagation through time, Adam, and portable NumPy weights."""
import numpy as np

BASE = {"length": 6, "units": 12, "lr": .003, "dropout": .10}


def init_params(n_features, hidden, rng):
    def weight(a, b):
        lim = np.sqrt(6 / (a + b))
        return rng.uniform(-lim, lim, (a, b)).astype("float32")
    return {"wx": weight(n_features, hidden), "wh": weight(hidden, hidden) * .65,
            "bh": np.zeros(hidden, "float32"), "wy": weight(hidden, 1).ravel(),
            "by": np.zeros(1, "float32")}


def hidden_states(x, p):
    states = [np.zeros((len(x), len(p["bh"])), dtype="float32")]
    for step in x.transpose(1, 0, 2):
        states.append(np.tanh(step @ p["wx"] + states[-1] @ p["wh"] + p["bh"]))
    return states


def predict(x, p):
    return hidden_states(x, p)[-1] @ p["wy"] + p["by"]


def fit(x, y, vx, vy, config, seed, epochs, patience):
    rng = np.random.default_rng(seed)
    p = init_params(x.shape[-1], config["units"], rng)
    m, v = ({key: np.zeros_like(val) for key, val in p.items()} for _ in range(2))
    history, best, best_loss = [], {}, np.inf
    step = stale = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(order), 128):
            ids = order[start:start + 128]
            batch = x[ids]
            states = hidden_states(batch, p)
            keep = 1 - config["dropout"]
            mask = (rng.random(states[-1].shape) < keep).astype("float32") / keep
            output = states[-1] * mask
            error = (2 / len(ids)) * (output @ p["wy"] + p["by"] - y[ids])
            grads = {"wy": output.T @ error, "by": np.atleast_1d(error.sum()),
                     "wx": np.zeros_like(p["wx"]), "wh": np.zeros_like(p["wh"]),
                     "bh": np.zeros_like(p["bh"])}
            dh = error[:, None] * p["wy"] * mask
            for t in reversed(range(batch.shape[1])):
                dz = dh * (1 - states[t + 1] ** 2)
                grads["wx"] += batch[:, t].T @ dz
                grads["wh"] += states[t].T @ dz
                grads["bh"] += dz.sum(axis=0)
                dh = dz @ p["wh"].T
            step += 1
            for key, grad in grads.items():
                grad = np.clip(grad, -5., 5.)
                m[key] = .9 * m[key] + .1 * grad
                v[key] = .999 * v[key] + .001 * grad ** 2
                p[key] -= config["lr"] * (m[key] / (1 - .9 ** step)) / (np.sqrt(v[key] / (1 - .999 ** step)) + 1e-8)
        loss = float(np.mean((predict(x, p) - y) ** 2))
        val_loss = float(np.mean((predict(vx, p) - vy) ** 2))
        history.append({"epoch": epoch, "train_mse_z": loss, "val_mse_z": val_loss})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best = {k: a.copy() for k, a in p.items()}
            best_epoch = epoch
        else:
            stale += 1
            if stale >= patience:
                break
    return best, history, best_epoch
