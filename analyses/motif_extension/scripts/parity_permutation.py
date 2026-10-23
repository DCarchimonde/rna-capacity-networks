"""Calibrate the sparse contrast with a finite-sample randomization test.

Uses the same simulated cell cohorts as parity_power.py (identical generation
seed and order). Labels are exchangeable under the specified identical-group
null. The statistic is the absolute difference of mean shared-capacity loss.
Both groups have the same population independent-edge flow B by construction,
so a loss contrast also implies a normalized competition-fraction contrast.
The tested statistic does not involve minimum-edge selection or a ratio.
"""

from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import beta
from parity_power import SEED, REPS, SCALES, DEPTHS, NS, K, generate, map_values, truth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PERM = 199


def run():
    gen = np.random.default_rng(SEED)
    perm = np.random.default_rng(26101904)
    rows = []
    for s in SCALES:
        for L in DEPTHS:
            tic = time.perf_counter()
            for setting, t0, t1 in [("alternative", 0.5, -0.5), ("null_same_population", 0.0, 0.0)]:
                X0 = generate(gen, s, L, t0, REPS * max(NS))
                X1 = generate(gen, s, L, t1, REPS * max(NS))
                V0 = map_values(X0, L)
                V1 = map_values(X1, L)
                a = np.stack([v[:, 0] for v in V0], axis=-1).reshape(REPS, max(NS), 2)
                b = np.stack([v[:, 0] for v in V1], axis=-1).reshape(REPS, max(NS), 2)
                for n in NS:
                    ps = []
                    zero = 0
                    for rep in range(REPS):
                        av = a[rep, :n]
                        bv = b[rep, :n]
                        pooled = np.concatenate([av, bv])
                        uniq, counts = np.unique(pooled, axis=0, return_counts=True)
                        obs = abs(av.mean(0) - bv.mean(0))
                        if len(uniq) == 1:
                            ps.append([1.0, 1.0])
                            zero += 1
                            continue
                        # Conditional allocation of categories to one group implements
                        # uniform random reassignment of n of the 2n observed cell labels.
                        allocation = perm.multivariate_hypergeometric(counts, n, size=PERM)
                        total = counts @ uniq
                        g0 = allocation @ uniq
                        stats = np.abs(2 * g0 - total) / n
                        tol = (
                            64
                            * np.finfo(float).eps
                            * np.maximum(np.maximum(stats.max(0), obs), 1e-300)
                        )
                        ps.append((1 + np.sum(stats >= obs - tol, axis=0)) / (PERM + 1))
                    ps = np.array(ps)
                    for j, method in enumerate(["conditional", "multinomial_plugin"]):
                        hits = int(np.sum(ps[:, j] <= 0.05))
                        lo = 0.0 if hits == 0 else beta.ppf(0.025, hits, REPS - hits + 1)
                        hi = 1.0 if hits == REPS else beta.ppf(0.975, hits + 1, REPS - hits)
                        rows.append(
                            dict(
                                s=s,
                                L=L,
                                n=n,
                                k=K,
                                setting=setting,
                                method=method,
                                repetitions=REPS,
                                permutations=PERM,
                                rejection_rate=hits / REPS,
                                rejection_ci_low=lo,
                                rejection_ci_high=hi,
                                constant_statistic_cohorts=zero,
                                molecular_budget_per_group=n * L,
                                true_raw_loss_difference=truth(s, t1)["loss"]
                                - truth(s, t0)["loss"],
                            )
                        )
            print(
                "permutation completed",
                s,
                L,
                "seconds",
                round(time.perf_counter() - tic, 2),
                flush=True,
            )
            pd.DataFrame(rows).to_csv(OUT / "parity_permutation_power.csv", index=False)
    spec = dict(
        generation_seed=SEED,
        permutation_seed=26101904,
        permutations=PERM,
        repetitions=REPS,
        finite_sample_null="identical cell-count distributions in the two groups; exchangeable cell labels",
        statistic="absolute difference of mean per-cell shared-capacity loss",
        pvalue="(1 + number of permuted statistics >= observed)/(199+1)",
        allocation="multivariate hypergeometric category allocation; both estimators share identical permutations",
        constant_statistics="p=1, retained in denominator",
        reason="the initial normal Wald diagnostic was inflated at small n; permutation results are primary",
    )
    (ROOT / "permutation_design.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("DONE permutation calibration", flush=True)


if __name__ == "__main__":
    run()
