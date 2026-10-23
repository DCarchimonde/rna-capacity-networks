"""Independent null calibration and evaluation for the fixed R500 contrast.

256 A/A pairs calibrate a two-sided 5% acceptance interval. Separate 256
A/A pairs estimate size; separate A/B pairs (sharing only evaluation A) estimate
power. The alternative data never set the rejection thresholds.
"""

from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, norm
from engine import calc, score, population

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(26101879)
R = 512
N = 3000
L = 10000
rows = []
for s in [1e-4, 3e-4, 1e-3]:
    J = np.empty((3, R, N), int)
    J[:2] = 2 * (rng.random((2, R, N)) < 0.75)
    J[2] = 1 + 2 * (rng.random((R, N)) < 0.25)
    p = np.full((3 * R * N, 5), s)
    p[:, 0] = s * J.reshape(-1)
    rem = np.full(len(p), L, int)
    prob = np.ones(len(p))
    X = np.empty((len(p), 5), np.int64)
    for g in range(5):
        X[:, g] = rng.binomial(rem, p[:, g] / prob)
        rem -= X[:, g]
        prob -= p[:, g]
    u, inv = np.unique(X, axis=0, return_inverse=True)
    del X, p, J, rem, prob
    f, _ = calc(u, L, [500])
    F = f[inv].reshape(3, R, N, 5)
    truth, _ = population(s, [500])
    gap = float(score(truth)[0, 0] - score(truth)[1, 0])
    for n in [100, 300, 1000, 3000]:
        r = score(F[:, :, :n].mean(2))
        null = r[0] - r[1]
        alt = r[0] - r[2]
        cal = null[:256]
        assert np.isfinite(cal).all()
        lo, hi = np.quantile(cal, [0.025, 0.975])
        nullrejected = (null[256:] < lo) | (null[256:] > hi)
        altrejected = (alt[256:] < lo) | (alt[256:] > hi)
        c = binomtest(int(altrejected.sum()), 256).proportion_ci()
        snr = gap / alt[256:].std(ddof=1)
        rows.append(
            dict(
                scale=s,
                depth=L,
                cells=n,
                k=500,
                calibration_pairs=256,
                evaluation_pairs=256,
                null_low=lo,
                null_high=hi,
                empirical_size=nullrejected.mean(),
                empirical_power=altrejected.mean(),
                power_ci_low=c.low,
                power_ci_high=c.high,
                true_gap=gap,
                finite_alt_fraction=np.isfinite(alt[256:]).mean(),
                normal_approx_cells_for_80_percent=n
                * ((norm.ppf(0.975) + norm.ppf(0.8)) / snr) ** 2,
            )
        )
    pd.DataFrame(rows).to_csv(ROOT / "results/high_order_detection_power.csv", index=False)
    print("power done", s, flush=True)
