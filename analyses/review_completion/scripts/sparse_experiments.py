"""Matched-moment simulations at realistic sparse abundance; no latent truth estimated from data."""

from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from engine import calc, score, population

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
spec = json.loads((ROOT / "config/analysis_spec.json").read_text())
ks = np.array(spec["resolutions"])
ns = spec["cells"]
R = spec["independent_replicates"]
N = max(ns)
rng = np.random.default_rng(spec["seed"])
rows = []
truthrows = []
timings = []
for si, s in enumerate(spec["sparse_scales"]):
    truth, bound = population(s, ks)
    tr = score(truth)
    for j, k in enumerate(ks):
        # Every component lies in [computed, computed+bound].
        lo = truth[:, j, 1:].min(1) / (truth[:, j, 0] + bound[:, j])
        hi = (truth[:, j, 1:].min(1) + bound[:, j]) / truth[:, j, 0]
        truthrows.append(
            dict(
                scale=s,
                k=k,
                R_A=tr[0, j],
                R_B=tr[1, j],
                difference=tr[0, j] - tr[1, j],
                difference_lower=lo[0] - hi[1],
                difference_upper=hi[0] - lo[1],
                max_component_tail_bound=bound[:, j].max(),
                mean_graph_fraction=5.5 * s,
            )
        )
    for L in spec["depths"]:
        start = time.perf_counter()
        J = np.empty((2, R, N), int)
        J[0] = 2 * (rng.random((R, N)) < 0.75)
        J[1] = 1 + 2 * (rng.random((R, N)) < 0.25)
        p = np.full((2 * R * N, 5), s)
        p[:, 0] = s * J.reshape(-1)
        rem = np.full(len(p), L, int)
        prob = np.ones(len(p))
        X = np.empty((len(p), 5), np.int64)
        for g in range(5):
            X[:, g] = rng.binomial(rem, p[:, g] / prob)
            rem -= X[:, g]
            prob -= p[:, g]
        u, inv = np.unique(X, axis=0, return_inverse=True)
        del X, p, rem, prob, J
        print("start", s, L, "unique", len(u), flush=True)
        for mode in [False, True]:
            method = "multinomial_plugin" if mode else "conditional"
            f, b = calc(u, L, ks, plugin=mode)
            F = f[inv].reshape(2, R, N, len(ks), 5)
            for n in ns:
                A = F[:, :, :n].mean(2)
                est = score(A)
                diff = est[0] - est[1]
                for j, k in enumerate(ks):
                    finite = np.isfinite(diff[:, j])
                    known = tr[0, j] - tr[1, j]
                    z = diff[finite, j]
                    successes = int(np.sum(z > 0)) if k > 3 else 0
                    ci = binomtest(successes, R).proportion_ci() if k > 3 else None
                    for bpop, name in enumerate(["A", "B"]):
                        r = est[bpop, :, j]
                        valid = np.isfinite(r)
                        rows.append(
                            dict(
                                scale=s,
                                depth=L,
                                cells=n,
                                k=k,
                                method=method,
                                population=name,
                                replicates=R,
                                true_R=tr[bpop, j],
                                true_difference=known,
                                finite_score_fraction=float(valid.mean()),
                                ratio_bias=(
                                    float(np.mean(r[valid]) - tr[bpop, j])
                                    if valid.any()
                                    else np.nan
                                ),
                                ratio_rmse=(
                                    float(np.sqrt(np.mean((r[valid] - tr[bpop, j]) ** 2)))
                                    if valid.any()
                                    else np.nan
                                ),
                                component_mse=float(np.mean((A[bpop, :, j] - truth[bpop, j]) ** 2)),
                                mean_estimated_difference=float(z.mean()) if len(z) else np.nan,
                                difference_sd=float(z.std(ddof=1)) if len(z) > 1 else np.nan,
                                true_gap_over_difference_sd=(
                                    float(known / z.std(ddof=1))
                                    if len(z) > 1 and z.std() > 0 and k > 3
                                    else np.nan
                                ),
                                strict_direction_accuracy=successes / R if k > 3 else np.nan,
                                direction_ci_low=ci.low if ci else np.nan,
                                direction_ci_high=ci.high if ci else np.nan,
                                finite_pair_fraction=float(finite.mean()),
                                max_plugin_component_bound=float(b.max()),
                            )
                        )
            del F, f
        timings.append(
            dict(
                scale=s,
                depth=L,
                generated_cells=2 * R * N,
                unique_counts=len(u),
                elapsed_seconds=time.perf_counter() - start,
            )
        )
        pd.DataFrame(rows).to_csv(OUT / "sparse_estimation.csv", index=False)
        pd.DataFrame(truthrows).to_csv(OUT / "sparse_truth.csv", index=False)
        pd.DataFrame(timings).to_csv(OUT / "sparse_runtime.csv", index=False)
        print("done", s, L, round(time.perf_counter() - start, 2), flush=True)
print("COMPLETE", len(rows), flush=True)
