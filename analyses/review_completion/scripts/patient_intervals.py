"""Within-patient joint component bootstrap, projected through min/ratio.

Intervals describe cell resampling in each patient and are distinct from
between-patient bootstrap intervals in the method comparison.
"""

from pathlib import Path
import numpy as np, pandas as pd
from engine import score

ROOT = Path(__file__).resolve().parents[1]
P = ROOT.parents[1]
OUT = ROOT / "results"
rng = np.random.default_rng(26101881)
rows = []
z = np.load(OUT / "expanded_selected_cellflows.npz")
old = np.load(P / "data/development_counts.npz")["patients"]
mask = np.isin(z["patients"][z["patient_index"]], old)
datasets = [("D", z["common_flows"][mask], z["patients"][z["patient_index"][mask]])]
zi = np.load(OUT / "independent_common_cellflows.npz")
datasets.append(("I", zi["flows"], zi["patient"]))
for co, F, ids in datasets:
    for pid in np.unique(ids):
        f = F[ids == pid]
        n = len(f)
        mu = f.mean(0)
        estimate = score(mu)
        flat = f.reshape(n, -1)
        sd = flat.std(0, ddof=1) / np.sqrt(n)
        bs = []
        for lo in range(0, 1000, 100):
            w = rng.multinomial(n, np.full(n, 1 / n), size=min(100, 1000 - lo))
            bs.append(w @ flat / n)
        bs = np.concatenate(bs)
        t = np.divide(abs(bs - mu.reshape(-1)), sd, out=np.zeros_like(bs), where=sd > 0).max(1)
        q = np.quantile(t, 0.95)
        lower = np.maximum(0, mu - q * sd.reshape(mu.shape))
        upper = mu + q * sd.reshape(mu.shape)
        rlo = np.divide(lower[:, 1:].min(1), upper[:, 0], out=np.zeros(5), where=upper[:, 0] > 0)
        rhi = np.divide(upper[:, 1:].min(1), lower[:, 0], out=np.ones(5), where=lower[:, 0] > 0)
        rhi = np.minimum(1, rhi)
        for j, k in enumerate([2, 8, 32, 128, 500]):
            rows.append(
                dict(
                    cohort=co,
                    patient=pid,
                    k=k,
                    R=estimate[j],
                    joint_component_95_low=rlo[j],
                    joint_component_95_high=rhi[j],
                    cells=n,
                    bootstrap_repetitions=1000,
                    interval_scope="simultaneous over the five resolutions and four component means within this patient; descriptive cell-bootstrap projection",
                )
            )
pd.DataFrame(rows).to_csv(OUT / "patient_score_intervals.csv", index=False)
print("intervals", len(rows), flush=True)
