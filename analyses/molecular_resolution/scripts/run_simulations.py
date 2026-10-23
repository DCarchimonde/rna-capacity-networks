"""Frozen controlled comparisons: exact and randomized estimators share R8."""

from pathlib import Path
from datetime import datetime, timezone
import time, json
import numpy as np
import pandas as pd
from resolution import conditional_flows, random_rarefaction, flows, r_from_flows, h_flows
from verify_theory import mean_matched_pop, second_moment_pop, population_numeric, population_exact
from analyze_cohorts import pair_counts

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/execution_spec.json").read_text())["simulations"]
METHODS = ["exact_conditional", "random_1", "random_8", "depth_plugin", "pair_only_R2"]


def to_arrays(pop):
    return np.array([float(w) for w, p in pop]), np.array([p for w, p in pop], float)


def draw_counts(p, l, rng):
    x = np.zeros_like(p, dtype=np.int64)
    remaining = l.copy()
    p_remaining = np.ones(len(l))
    for j in range(5):
        x[:, j] = rng.binomial(remaining, np.clip(p[:, j] / p_remaining, 0, 1))
        remaining -= x[:, j]
        p_remaining -= p[:, j]
    return x


def agg(z, patient, typ, n):
    out = np.zeros((n, z.shape[1]))
    for i in range(n):
        out[i] = 0.5 * z[(patient == i) & (typ == 0)].mean(0) + 0.5 * z[
            (patient == i) & (typ == 1)
        ].mean(0)
    return out


def known_flows(family, k, eta):
    if family == "identical_biology_depth_gradient":
        p = np.array([0.06, 0.08, 0.10, 0.03, 0.05])
        return np.tile(population_numeric([(1, p)], k), (len(eta), 1))
    pa, pb = (
        (mean_matched_pop(0), mean_matched_pop(1))
        if family == "mean_matched_joint"
        else (second_moment_pop(0), second_moment_pop(1))
    )
    if k <= 4:
        qa, qb = population_exact(pa, k), population_exact(pb, k)
        aa, bb = np.array(qa, float), np.array(qb, float)
        if qa == qb:
            return np.tile(aa, (len(eta), 1))
    else:
        aa, bb = population_numeric(pa, k), population_numeric(pb, k)
    return (1 - eta[:, None]) * aa + eta[:, None] * bb


def main():
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    n = SPEC["patients"]
    ne, nf = SPEC["Epi"], SPEC["Fibro"]
    m = ne + nf
    k = SPEC["k"]
    eta = np.linspace(0.1, 0.9, n)
    patient = np.repeat(np.arange(n), m)
    typ = np.tile(np.r_[np.zeros(ne, int), np.ones(nf, int)], n)
    half = np.tile(np.r_[np.arange(ne) % 2, np.arange(nf) % 2], n)
    truth = {f: known_flows(f, k, eta) for f in SPEC["families"]}
    truth_rows = []
    for family in SPEC["families"]:
        t2 = known_flows(family, 2, eta)
        for i in range(n):
            truth_rows.append(
                {
                    "family": family,
                    "patient": i,
                    "eta": eta[i],
                    "true_R8": r_from_flows(truth[family][i]),
                    "true_R2": r_from_flows(t2[i]),
                }
            )
    pd.DataFrame(truth_rows).to_csv(ROOT / "results/simulation_truth.csv", index=False)
    rows = []
    patient_rows = []
    conditional = []
    maxnorm = 0.0
    for fi, family in enumerate(SPEC["families"]):
        for di, L in enumerate(SPEC["depths"]):
            for rep in range(SPEC["cohort_replicates"]):
                seed = SPEC["seed_base"] + fi * 100000 + di * 1000 + rep
                rng = np.random.default_rng(seed)
                if family == "identical_biology_depth_gradient":
                    latent = np.tile([0.06, 0.08, 0.1, 0.03, 0.05], (len(patient), 1))
                    depth_by_patient = np.rint(np.geomspace(L, 4 * L, n)).astype(np.int64)
                else:
                    pa, pb = (
                        (mean_matched_pop(0), mean_matched_pop(1))
                        if family == "mean_matched_joint"
                        else (second_moment_pop(0), second_moment_pop(1))
                    )
                    wa, aa = to_arrays(pa)
                    wb, bb = to_arrays(pb)
                    choose_b = rng.random(len(patient)) < eta[patient]
                    latent = np.empty((len(patient), 5))
                    for flag, ww, states in [(False, wa, aa), (True, wb, bb)]:
                        keep = choose_b == flag
                        latent[keep] = states[rng.choice(len(ww), keep.sum(), p=ww)]
                    depth_by_patient = np.full(n, L, dtype=np.int64)
                depth = depth_by_patient[patient]
                x = draw_counts(latent, depth, rng)
                exact, err = conditional_flows(x, depth, [k])
                exact = exact[:, 0]
                maxnorm = max(maxnorm, err.max())
                random1 = random_rarefaction(x, depth, k, np.random.default_rng(seed + 30000000))
                random8 = random_rarefaction(
                    x, depth, k, np.random.default_rng(seed + 40000000), draws=8
                )
                plugin = flows(x / depth[:, None])
                pair = h_flows(x, depth)
                target = truth[family]
                trueR = r_from_flows(target)
                isnull = family == "identical_biology_depth_gradient"
                for method, z in zip(METHODS, [exact, random1, random8, plugin, pair]):
                    af = agg(z, patient, typ, n)
                    rr = r_from_flows(af)
                    r0 = r_from_flows(agg(z[half == 0], patient[half == 0], typ[half == 0], n))
                    r1 = r_from_flows(agg(z[half == 1], patient[half == 1], typ[half == 1], n))
                    retained = pair_counts(r0, r1)
                    accuracy = (
                        pair_counts(trueR, rr)["all_pair_retention"] if not isnull else np.nan
                    )
                    rows.append(
                        {
                            "family": family,
                            "base_depth": L,
                            "rep": rep,
                            "method": method,
                            "flow_MSE_R8_target": (
                                np.mean((af - target) ** 2) if method != "pair_only_R2" else np.nan
                            ),
                            "R8_MSE": np.nanmean((rr - trueR) ** 2),
                            "R8_pair_accuracy": accuracy,
                            "split_all_pair_retention": retained["all_pair_retention"],
                            "split_reference_resolution": retained["reference_resolution"],
                            "finite_patients": int(np.isfinite(rr).sum()),
                            "R_depth_correlation": (
                                np.corrcoef(depth_by_patient, rr)[0, 1]
                                if isnull and np.isfinite(rr).all()
                                else np.nan
                            ),
                        }
                    )
                    for i in range(n):
                        patient_rows.append(
                            {
                                "family": family,
                                "base_depth": L,
                                "rep": rep,
                                "method": method,
                                "patient": i,
                                "depth": depth_by_patient[i],
                                "R": rr[i],
                                "true_R8": trueR[i],
                                "A0": af[i, 0],
                                "A0_true": target[i, 0],
                            }
                        )
                if rep == 0:
                    # Conditional molecular randomization only; biology and
                    # original observed counts are held fixed.
                    ii = np.arange(min(300, len(depth)))
                    samples = np.stack(
                        [
                            random_rarefaction(
                                x[ii], depth[ii], k, np.random.default_rng(seed + 50000000 + q)
                            )
                            for q in range(256)
                        ]
                    )
                    var = samples.var(0, ddof=1).mean()
                    errmean = np.mean((samples.mean(0) - exact[ii]) ** 2)
                    conditional.append(
                        {
                            "family": family,
                            "base_depth": L,
                            "cells": len(ii),
                            "random_1_conditional_variance": var,
                            "random_8_conditional_variance_theoretical": var / 8,
                            "exact_conditional_randomization_variance": 0,
                            "MSE_256_draw_mean_vs_exact": errmean,
                        }
                    )
            print(
                json.dumps(
                    {"done": family, "base_depth": L, "replicates": SPEC["cohort_replicates"]}
                ),
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/simulation_replicates.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(ROOT / "results/simulation_patient_scores.csv", index=False)
    pd.DataFrame(conditional).to_csv(ROOT / "results/conditional_randomization.csv", index=False)
    endpoints = [
        "flow_MSE_R8_target",
        "R8_MSE",
        "R8_pair_accuracy",
        "split_all_pair_retention",
        "split_reference_resolution",
        "finite_patients",
        "R_depth_correlation",
    ]
    grouped = (
        df.groupby(["family", "base_depth", "method"], sort=False)[endpoints].mean().reset_index()
    )
    grouped.to_csv(ROOT / "results/simulation_summary.csv", index=False)
    gains = []
    for (family, L), group in df.groupby(["family", "base_depth"]):
        for method in ["random_1", "random_8"]:
            a = group[group.method == "exact_conditional"].set_index("rep")
            b = group[group.method == method].set_index("rep")
            d = (a.R8_pair_accuracy - b.R8_pair_accuracy).dropna().to_numpy()
            # Paired simulation-cohort bootstrap; descriptive Monte Carlo CI.
            if len(d):
                rng = np.random.default_rng(2610105288)
                draws = rng.integers(len(d), size=(2000, len(d)))
                lo, hi = np.quantile(d[draws].mean(1), [0.025, 0.975])
                gains.append(
                    {
                        "family": family,
                        "base_depth": L,
                        "baseline": method,
                        "mean_pair_accuracy_gain": d.mean(),
                        "MC_95_low": lo,
                        "MC_95_high": hi,
                    }
                )
    pd.DataFrame(gains).to_csv(ROOT / "results/simulation_paired_gains.csv", index=False)
    result = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic() - start,
        "cohorts": len(SPEC["families"]) * len(SPEC["depths"]) * SPEC["cohort_replicates"],
        "cells_per_cohort": len(patient),
        "max_hypergeom_normalization_error": float(maxnorm),
        "methods": METHODS,
        "fixed_k": k,
    }
    (ROOT / "results/simulation_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(grouped.to_string(index=False))
    print(pd.DataFrame(gains).to_string(index=False))


if __name__ == "__main__":
    main()
