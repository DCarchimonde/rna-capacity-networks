from pathlib import Path
import json
import numpy as np
import pandas as pd
from engine import calc, score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
rng = np.random.default_rng(26101874)
rows = []
rankrows = []
R = 512
N = 3000
ns = [100, 300, 1000, 3000]
for L in [1000, 10000]:
    estimates = {}
    for delta in [0, 0.02, 0.1, 0.5]:
        p = np.array([1e-4, 1e-4 * (1 + delta), 1e-4 * (1 - delta)])
        x = rng.multinomial(L, np.r_[p, 1 - p.sum()], size=(R, N))[..., :3]
        h = x[:, :, :1] * x[:, :, 1:] / (L * (L - 1))
        truth = p[0] * p[1:]
        truthR = truth.min() / truth.sum()
        estimates[delta] = {}
        for n in ns:
            H = h[:, :n].mean(1)
            den = H.sum(1)
            num = H.min(1)
            r = np.divide(num, den, out=np.full(R, np.nan), where=den > 0)
            estimates[delta][n] = r
            bootstrap_valid = 0
            covered = 0
            interval_width = []
            projected_covered = 0
            projected_width = []
            # Descriptive within-sample percentile intervals, measured against known truth.
            for rep in range(128):
                vals, count = np.unique(h[rep, :n], axis=0, return_counts=True)
                w = rng.multinomial(n, count / n, size=399)
                hb = w @ vals / n
                db = hb.sum(1)
                rb = np.divide(hb.min(1), db, out=np.full(399, np.nan), where=db > 0)
                if np.isfinite(rb).all():
                    lo, hi = np.quantile(rb, [0.025, 0.975])
                    bootstrap_valid += 1
                    covered += lo <= truthR <= hi
                    interval_width.append(hi - lo)
                # Bootstrap the smooth component means jointly, then project the rectangle
                # through min/(sum), including .5 whenever the component intervals overlap.
                center = h[rep, :n].mean(0)
                q = np.quantile(np.max(abs(hb - center), axis=1), 0.95)
                lower = np.maximum(0, center - q)
                upper = center + q
                if upper.sum() == 0:
                    lo, hi = 0, 0.5
                else:
                    lo = min(
                        lower[0] / (lower[0] + upper[1]) if lower[0] + upper[1] > 0 else 0,
                        lower[1] / (lower[1] + upper[0]) if lower[1] + upper[0] > 0 else 0,
                    )
                    hi = (
                        0.5
                        if max(lower) <= min(upper)
                        else min(upper[0] / (upper[0] + lower[1]), upper[1] / (upper[1] + lower[0]))
                    )
                projected_covered += lo <= truthR <= hi
                projected_width.append(hi - lo)
            rows.append(
                dict(
                    depth=L,
                    delta=delta,
                    cells=n,
                    replicates=R,
                    true_R=truthR,
                    finite_fraction=np.isfinite(r).mean(),
                    mean_R=np.nanmean(r),
                    ratio_bias=np.nanmean(r) - truthR,
                    numerator_bias_over_true_baseline=(num.mean() - truth.min()) / truth.sum(),
                    wrong_edge_fraction=np.mean(H[:, 0] < H[:, 1]) if delta > 0 else np.nan,
                    edge_tie_fraction=np.mean(H[:, 0] == H[:, 1]),
                    percentile_interval_valid=bootstrap_valid,
                    percentile_interval_attempts=128,
                    percentile_interval_coverage=(
                        covered / bootstrap_valid if bootstrap_valid else np.nan
                    ),
                    percentile_interval_mean_width=(
                        np.mean(interval_width) if interval_width else np.nan
                    ),
                    joint_component_projection_coverage=projected_covered / 128,
                    joint_component_projection_mean_width=np.mean(projected_width),
                )
            )
    for n in ns:
        for a, b in [(0, 0.02), (0, 0.1), (0.02, 0.1), (0.1, 0.5)]:
            ra, rb = estimates[a][n], estimates[b][n]
            rankrows.append(
                dict(
                    depth=L,
                    cells=n,
                    delta_A=a,
                    delta_B=b,
                    true_gap=(b - a) / 2,
                    strict_direction_accuracy=np.mean(ra > rb),
                    finite_fraction=np.mean(np.isfinite(ra) & np.isfinite(rb)),
                )
            )
pd.DataFrame(rows).to_csv(OUT / "near_tie_bias.csv", index=False)
pd.DataFrame(rankrows).to_csv(OUT / "near_tie_rank_accuracy.csv", index=False)
# Fixed panel composition, changing only its fraction of total RNA.
bg = []
q = np.array([0.15, 0.25, 0.30, 0.1, 0.2])
ks = [2, 8, 32, 128, 500]
for rho in np.geomspace(1e-5, 0.1, 41):
    f, b = calc(np.zeros((1, 5), int), [500], ks, True, P=(rho * q)[None, :])
    r = score(f)[0]
    for j, k in enumerate(ks):
        bg.append(dict(rho=rho, k=k, R=r[j], A0=f[0, j, 0], component_bound=b[0, j]))
pd.DataFrame(bg).to_csv(OUT / "background_only_truth.csv", index=False)
# Panel-conditioned population target via a homogeneous panel: integrate at rho=.999999
# using the archived exact multinomial implementation (n<=8).
import sys

sys.path.insert(0, str(ROOT.parent / "strong_comparators/scripts"))
from plugin import multinomial_flows

panel = []
for n in [2, 3, 4, 8]:
    f, _ = multinomial_flows(q[None, :], n)
    panel.append(dict(panel_resolution=n, R=score(f)[0]))
pd.DataFrame(panel).to_csv(OUT / "panel_background_invariant_truth.csv", index=False)
print(pd.DataFrame(rows).query("cells==3000").to_string(index=False), flush=True)
