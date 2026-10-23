"""Independent, fixed-budget numerical refinement for five close-rank patients.

Selected because three pair differences were within three recorded numerical
Monte Carlo SEs in the first decomposition. This is computational refinement
of already used data, not additional biological validation.
"""

from pathlib import Path
import sys, time, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
B = PROJECT / "analyses/review_completion"
sys.path.insert(0, str(B / "scripts"))
from general_graph import integrate

z = np.load(B / "inputs/atlas_integrins_all_cells.npz")
genes = list(z["genes"])
E = pd.read_csv(B / "inputs/integrin_edges_with_provenance.csv")
edges = np.array(
    [
        (genes.index(u), genes.index(v))
        for u, v in E[["alpha", "beta"]].itertuples(index=False, name=None)
    ]
)
ref = np.load(B / "results/expanded_fullgraph_cellflows.npz")
patients = ref["patients"]
take = (
    (z["tissue"] == "T")
    & (z["celltype"] == "Epi")
    & (z["depth"] >= 500)
    & np.isin(z["patient"], patients)
)
X = z["counts"][take]
L = z["depth"][take]
p = z["patient"][take]
selected = ["C132", "C169", "C171", "C156", "C168"]
draw_budget = {"C169": 16384, "C168": 16384}
indices = []
flows = []
records = []
for i, pid in enumerate(selected):
    ix = np.flatnonzero(p == pid)
    t = time.perf_counter()
    draws = draw_budget.get(pid, 1024)
    f, _ = integrate(X[ix], L[ix], 500, edges, batches=32, draws=draws, seed=261020100 + i)
    batch = f.mean(0)
    r = batch[:, 1:].min(1) / batch[:, 0]
    average = batch.mean(0)
    point = average[1:].min() / average[0]
    indices.extend(ix.tolist())
    flows.append(f.mean(1))
    rec = dict(
        patient=pid,
        cells=len(ix),
        R=float(point),
        integration_mc_se=float(r.std(ddof=1) / np.sqrt(len(r))),
        seconds=time.perf_counter() - t,
        batches=32,
        draws_per_batch=draws,
        seed=261020100 + i,
    )
    records.append(rec)
    print(rec, flush=True)
np.savez_compressed(
    ROOT / "results/refined_close_cellflows.npz",
    indices=np.array(indices),
    flows=np.concatenate(flows),
    k=500,
)
(ROOT / "results/refined_close_orders.json").write_text(json.dumps(records, indent=2) + "\n")
