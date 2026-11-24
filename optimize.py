"""Validation-only evolutionary searches. One search per seed with cumulative logs."""
import numpy as np

from air_data import FEATURES
from recurrent import BASE

POP, GENERATIONS = 6, 5
PARTICLES, ROUNDS = 5, 5
SEEDS = (7, 19, 42)
PENALTY = .03
LENGTHS, UNITS = (4, 6, 8), (8, 12, 16, 20, 24)


def genetic(score, seed):
    rng = np.random.default_rng(seed)
    n = len(FEATURES)
    pop = rng.integers(0, 2, (POP, n)).astype(bool)
    pop[0] = True
    history, records = [], []
    for gen in range(1, GENERATIONS + 1):
        for mask in pop:
            if not mask.any():
                mask[rng.integers(n)] = True
        fitness = []
        for mask in pop:
            error = score(tuple(np.flatnonzero(mask)), BASE)
            cost = error + PENALTY * mask.sum() / n
            fitness.append(cost)
            records.append({"generation": gen, "bits": "".join(map(str, mask.astype(int))),
                            "val_rmse": error, "fitness": float(cost)})
        elite = pop[np.argmin(fitness)].copy()
        history.append(float(min(fitness)))
        new = [elite]
        def tournament():
            ids = rng.choice(POP, size=3, replace=False)
            return pop[ids[np.argmin(np.asarray(fitness)[ids])]]
        while len(new) < POP:
            left, right = tournament(), tournament()
            child = left.copy()
            if rng.random() < .8:
                cut = rng.integers(1, n)
                child[cut:] = right[cut:]
            child ^= rng.random(n) < 1 / n
            new.append(child)
        pop = np.asarray(new)
    return {"seed": seed, "features": [FEATURES[i] for i in np.flatnonzero(elite)],
            "fitness": history[-1], "history": history, "candidates": records}


def decode(position):
    length, units, rate, drop = position
    return {"length": LENGTHS[round(float(length) * (len(LENGTHS) - 1))],
            "units": UNITS[round(float(units) * (len(UNITS) - 1))],
            "lr": round(float(10 ** (-3 + rate)), 5), "dropout": round(.25 * float(drop), 2)}


def swarm(score, indices, seed):
    rng = np.random.default_rng(seed)
    pos = rng.random((PARTICLES, 4))
    vel = rng.uniform(-.15, .15, size=pos.shape)
    pbest, p_cost = pos.copy(), np.full(PARTICLES, np.inf)
    history, records = [], []
    for iteration in range(1, ROUNDS + 1):
        for i, particle in enumerate(pos):
            params = decode(particle)
            cost = score(indices, params)
            records.append({"iteration": iteration, "particle": i + 1, "params": params,
                            "position": particle.tolist(), "val_rmse": cost})
            if cost < p_cost[i]:
                p_cost[i], pbest[i] = cost, particle.copy()
        best = pbest[np.argmin(p_cost)].copy()
        history.append(float(np.min(p_cost)))
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = np.clip(.65 * vel + 1.4 * r1 * (pbest - pos) + 1.4 * r2 * (best - pos), -.35, .35)
        pos = np.clip(pos + vel, 0, 1)
    return {"seed": seed, "parameters": decode(best), "fitness": history[-1],
            "history": history, "candidates": records}
