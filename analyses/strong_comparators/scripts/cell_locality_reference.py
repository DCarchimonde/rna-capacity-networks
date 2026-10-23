"""Five-gene RNA targets. No protein/assembly quantities are inferred."""

import numpy as np
from scipy.stats import rankdata

EDGES = [(0, 2), (1, 2), (1, 3), (1, 4)]
EDGE_NAMES = ["ITGA5_ITGB1", "ITGAV_ITGB1", "ITGAV_ITGB3", "ITGAV_ITGB6"]
H_METHODS = [
    "mixed_means",
    "typed_means",
    "typed_crosscell",
    "cell_fraction_product",
    "cell_factorial",
]
R_METHODS = ["pooled_R", "typed_R", "observed_cell_R"]


def flows(p):
    """F0, and four F-minus-edge maximum flows, with graph held fixed."""
    a, v, b, c, d = np.moveaxis(np.asarray(p, float), -1, 0)
    f0 = np.minimum.reduce([a + v, v + b, b + c + d])
    fab = np.minimum(v, b + c + d)
    fvb = np.minimum.reduce([a + v, a + c + d, b + v, b + c + d])
    fvc = np.minimum.reduce([a + v, v + b, b + d])
    fvd = np.minimum.reduce([a + v, v + b, b + c])
    return np.stack([f0, fab, fvb, fvc, fvd], -1)


def resilience(f):
    f = np.asarray(f, float)
    return np.divide(
        f[..., 1:].min(-1), f[..., 0], out=np.full(f.shape[:-1], np.nan), where=f[..., 0] > 0
    )


def factorial_products(x, l):
    x = np.asarray(x, float)
    l = np.asarray(l, float)
    if np.any(l < 2):
        raise ValueError("At least two UMIs per included cell required.")
    return np.stack([x[..., i] * x[..., j] / (l * (l - 1)) for i, j in EDGES], -1)


def summarize_type(x, l):
    x = np.asarray(x, float)
    l = np.asarray(l, float)
    n = len(l)
    if n < 2:
        raise ValueError("At least two cells per type required for cross-cell U statistic.")
    y = x / l[:, None]
    mu = y.mean(0)
    prod = np.stack([y[:, i] * y[:, j] for i, j in EDGES], -1)
    cross = np.array(
        [
            (y[:, i].sum() * y[:, j].sum() - prod[:, k].sum()) / (n * (n - 1))
            for k, (i, j) in enumerate(EDGES)
        ]
    )
    h = factorial_products(x, l)
    return {
        "means": mu,
        "mean_products": np.array([mu[i] * mu[j] for i, j in EDGES]),
        "crosscell": cross,
        "naive": prod.mean(0),
        "factorial": h.mean(0),
        "cell_flows": flows(y).mean(0),
        "mean_flows": flows(mu),
        "positive_pair_cells": np.array([((x[:, i] > 0) & (x[:, j] > 0)).sum() for i, j in EDGES]),
        "percell_factorial": h,
    }


def summarize_patient(pairs):
    t = [summarize_type(*p) for p in pairs]
    mu = np.array([a["means"] for a in t])
    mix = mu.mean(0)
    hh = np.stack(
        [
            np.array([mix[i] * mix[j] for i, j in EDGES]),
            np.mean([a["mean_products"] for a in t], 0),
            np.mean([a["crosscell"] for a in t], 0),
            np.mean([a["naive"] for a in t], 0),
            np.mean([a["factorial"] for a in t], 0),
        ]
    )
    ff = np.stack(
        [
            flows(mix),
            np.mean([a["mean_flows"] for a in t], 0),
            np.mean([a["cell_flows"] for a in t], 0),
        ]
    )
    return hh, resilience(ff), ff, t


def pair_metrics(ref, query):
    """All-pair strict retention counts query ties as unsuccessful."""
    ref = np.asarray(ref, float)
    query = np.asarray(query, float)
    n = len(ref)
    i, j = np.triu_indices(n, 1)
    good = np.isfinite(ref[i]) & np.isfinite(ref[j]) & (ref[i] != ref[j])
    qgood = np.isfinite(query[i]) & np.isfinite(query[j])
    kept = good & qgood & ((query[i] - query[j]) * (ref[i] - ref[j]) > 0)
    reversed_ = good & qgood & ((query[i] - query[j]) * (ref[i] - ref[j]) < 0)
    finite = np.isfinite(ref) & np.isfinite(query)
    if finite.sum() > 1:
        d = np.abs(
            rankdata(ref[finite], method="average") - rankdata(query[finite], method="average")
        ).sum() / (finite.sum() * (finite.sum() - 1))
    else:
        d = np.nan
    return {
        "D": float(d),
        "reference_resolution": float(good.mean()),
        "query_resolution": float((qgood & (query[i] != query[j])).mean()),
        "strict_accuracy": float(kept.sum() / good.sum()) if good.any() else np.nan,
        "all_pair_retention": float(kept.mean()),
        "reversal_fraction": float(reversed_.mean()),
        "finite_patients": int(finite.sum()),
        "strict_reference_pairs": int(good.sum()),
    }


def population_states(rho, delta, scale=0.0002):
    """Five states with identical per-patient means for every rho."""
    mu = np.array([1.0, 1.0 + delta, 1.0, 1.0, 1.0]) * scale
    states = np.zeros((5, 5))
    states[0] = mu
    states[1, [0, 2]] = 4 * scale
    states[2, 1] = 4 * (1 + delta) * scale
    states[3, 3] = 4 * scale
    states[4, 4] = 4 * scale
    weights = np.array([1 - rho, rho / 4, rho / 4, rho / 4, rho / 4])
    assert np.allclose(weights @ states, mu, atol=1e-18, rtol=0)
    return states, weights


def true_patient(rho, delta, signal_type):
    states = []
    weights = []
    for t in range(2):
        s, w = population_states(rho, delta) if t == signal_type else population_states(0.0, 0.0)
        states.append(s)
        weights.append(w)
    # Use the exact specified mean; do not manufacture near-ties from summation roundoff.
    mus = np.array([s[0] for s in states])
    mix = mus.mean(0)
    hs = np.array(
        [w @ np.array([s[:, i] * s[:, j] for i, j in EDGES]).T for s, w in zip(states, weights)]
    )
    fs = np.array([w @ flows(s) for s, w in zip(states, weights)])
    f = np.stack([flows(mix), flows(mus).mean(0), fs.mean(0)])
    return hs.mean(0), resilience(f), f, states, weights
