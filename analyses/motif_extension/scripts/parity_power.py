"""Sparse fixed-panel experiment: identical complete bivariate marginals.

Three signal genes have p_i=s*(0.1+Z_i), with three inert complementary
genes s*(1.1-Z_i), keeping the panel fraction 3.6s constant. All eight
binary states have positive mass: Pr(Z=z)=(1+theta*(-1)^sum(z))/8.
The alternative compares theta=+0.5 and -0.5; the null compares theta=0
with itself. Cells are iid. k=500 is fixed before results are inspected.
The test compares a smooth normalized shared-capacity deficit, not R_k.
"""

from pathlib import Path
from itertools import product
import json, time
import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from pair_flows import integrate_pairs, multinomial_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 26101903
REPS = 256
SCALES = [1e-5, 1e-4, 3e-4, 1e-3]
DEPTHS = [1000, 10000]
NS = [100, 300, 1000, 3000]
K = 500
BITS = np.array(list(product([0, 1], repeat=3)))


def truth(s, theta, k=K):
    ps = s * (0.1 + BITS)
    w = (1 + theta * (-1.0) ** BITS.sum(1)) / 8
    b1 = multinomial_pairs(ps[:, [0, 1]], k)
    b2 = multinomial_pairs(ps[:, [0, 2]], k)
    a = multinomial_pairs(np.column_stack([ps[:, 0], ps[:, 1] + ps[:, 2]]), k)
    B = w @ (b1 + b2)
    A = w @ a
    return dict(R=min(w @ b1, w @ b2) / A, C=(B - A) / B, B=B, A=A, loss=B - A)


def generate(rng, s, L, theta, N):
    w = (1 + theta * (-1.0) ** BITS.sum(1)) / 8
    state = rng.choice(8, size=N, p=w)
    out = np.empty((N, 3), dtype=np.int64)
    for j, bits in enumerate(BITS):
        ix = np.flatnonzero(state == j)
        p = s * (0.1 + bits)
        out[ix] = rng.multinomial(L, np.r_[p, 1 - p.sum()], size=len(ix))[:, :3]
    return out


def map_values(X, L):
    unique, inverse = np.unique(X, axis=0, return_inverse=True)
    # Batch all pair counts, deduplicating the two leaf labels and their sum.
    counts = np.concatenate(
        [
            unique[:, [0, 1]],
            unique[:, [0, 2]],
            np.column_stack([unique[:, 0], unique[:, 1] + unique[:, 2]]),
        ]
    )
    pair_unique, pair_inverse = np.unique(np.sort(counts, axis=1), axis=0, return_inverse=True)
    conditional = integrate_pairs(
        pair_unique, np.full(len(pair_unique), L), np.array([[0, 1]]), [K]
    )[:, 0, 0]
    plugin = multinomial_pairs(pair_unique / L, K)

    def expand(values):
        v = values[pair_inverse].reshape(3, len(unique)).T
        B = v[:, 0] + v[:, 1]
        A = v[:, 2]
        C = B - A
        C[np.any(unique == 0, axis=1)] = 0.0
        assert C.min() > -1e-12
        return np.column_stack([np.maximum(C, 0.0), B, A, np.minimum(v[:, 0], v[:, 1])])[inverse]

    return expand(conditional), expand(plugin)


def estimate(Y, n):
    y = Y[:, :n, :]
    m = y.mean(1)
    C = np.divide(m[:, 0], m[:, 1], out=np.full(len(m), np.nan), where=m[:, 1] > 0)
    z = y[:, :, 0] - C[:, None] * y[:, :, 1]
    se = np.sqrt(np.var(z, axis=1, ddof=1) / n) / m[:, 1]
    return C, se


def run():
    spec = dict(
        seed=SEED,
        repetitions=REPS,
        scales=SCALES,
        depths=DEPTHS,
        sample_sizes=NS,
        k=K,
        baseline_expression=".1*s",
        constant_six_gene_fraction="3.6*s",
        alternative_theta=[0.5, -0.5],
        null_theta=[0, 0],
        test="two-sided normal Wald test on difference of normalized shared-capacity deficits using cell influence variances",
        undefined_test="count as not rejected; report its frequency",
        estimators=["conditional hypergeometric", "multinomial plug-in"],
        status="new exploratory simulation design; no clinical endpoint is used",
    )
    (ROOT / "power_design.json").write_text(json.dumps(spec, indent=2) + "\n")
    rng = np.random.default_rng(SEED)
    rows = []
    truthrows = []
    for s in SCALES:
        for L in DEPTHS:
            t = time.perf_counter()
            for label, theta0, theta1 in [("alternative", 0.5, -0.5), ("null", 0.0, 0.0)]:
                tr0 = truth(s, theta0)
                tr1 = truth(s, theta1)
                truthrows.append(
                    dict(
                        s=s,
                        L=L,
                        setting=label,
                        C0=tr0["C"],
                        C1=tr1["C"],
                        R0=tr0["R"],
                        R1=tr1["R"],
                        B=tr0["B"],
                        truth_difference_C=tr1["C"] - tr0["C"],
                        truth_difference_R=tr1["R"] - tr0["R"],
                    )
                )
                X0 = generate(rng, s, L, theta0, REPS * max(NS))
                X1 = generate(rng, s, L, theta1, REPS * max(NS))
                V0 = map_values(X0, L)
                V1 = map_values(X1, L)
                for method, a, b in zip(["conditional", "multinomial_plugin"], V0, V1):
                    a = a.reshape(REPS, max(NS), 4)
                    b = b.reshape(REPS, max(NS), 4)
                    for n in NS:
                        e0, se0 = estimate(a, n)
                        e1, se1 = estimate(b, n)
                        delta = e1 - e0
                        se = np.hypot(se0, se1)
                        valid = np.isfinite(delta) & np.isfinite(se) & (se > 0)
                        reject = valid & (np.abs(delta) > norm.ppf(0.975) * se)
                        nreject = int(reject.sum())
                        lower = (
                            0.0 if nreject == 0 else beta.ppf(0.025, nreject, REPS - nreject + 1)
                        )
                        upper = (
                            1.0 if nreject == REPS else beta.ppf(0.975, nreject + 1, REPS - nreject)
                        )
                        target = tr1["C"] - tr0["C"]
                        defined = np.isfinite(delta)
                        covered = valid & (np.abs(delta - target) <= norm.ppf(0.975) * se)
                        rows.append(
                            dict(
                                s=s,
                                L=L,
                                n=n,
                                k=K,
                                setting=label,
                                method=method,
                                repetitions=REPS,
                                rejection_rate=float(reject.mean()),
                                rejection_ci_low=float(lower),
                                rejection_ci_high=float(upper),
                                undefined_tests=int((~valid).sum()),
                                defined_estimates=int(defined.sum()),
                                truth_difference=target,
                                mean_difference=(
                                    float(delta[defined].mean()) if defined.any() else np.nan
                                ),
                                mse=(
                                    float(np.mean((delta[defined] - target) ** 2))
                                    if defined.any()
                                    else np.nan
                                ),
                                wald_coverage=float(covered.mean()),
                            )
                        )
            print("completed", s, L, "seconds", round(time.perf_counter() - t, 2), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "parity_power.csv", index=False)
            pd.DataFrame(truthrows).to_csv(OUT / "parity_sparse_truth.csv", index=False)
    # Whole resolution curves, including the exact even/odd endpoints.
    curves = []
    for s in SCALES:
        for theta in [1.0, 0.5, 0.0, -0.5, -1.0]:
            for k in [2, 3, 8, 32, 128, 500, 2000]:
                t = truth(s, theta, k)
                curves.append(dict(s=s, theta=theta, k=k, **t))
    pd.DataFrame(curves).to_csv(OUT / "parity_resolution_curves.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
