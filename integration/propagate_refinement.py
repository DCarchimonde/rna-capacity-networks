"""Propagate the archived fresh-draw refinement into all dependent score tables.

Idempotent. It does not rerun or tune the numerical integration.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analyses/review_completion/results"
EXT = ROOT / "analyses/motif_extension/results"
z = np.load(OUT / "expanded_fullgraph_cellflows.npz")
data = {k: z[k] for k in z.files}
r = np.load(EXT / "refined_close_cellflows.npz")
ik = int(np.flatnonzero(data["ks"] == int(r["k"]))[0])
data["flows"][r["indices"], ik, :] = r["flows"]
np.savez_compressed(OUT / "expanded_fullgraph_cellflows.npz", **data)
ids = data["patients"][data["patient_index"]]
refined = json.loads((EXT / "refined_close_orders.json").read_text())
scores = pd.read_csv(OUT / "expanded_graph_scores.csv")
edges = pd.read_csv(ROOT / "analyses/review_completion/inputs/integrin_edges_with_provenance.csv")
audit = []
for item in refined:
    pid = item["patient"]
    means = data["flows"][ids == pid, ik, :].mean(axis=0)
    val = means[1:].min() / means[0]
    assert abs(val - item["R"]) < 1e-12
    mask = (scores.graph == "integrin26") & (scores.patient == pid) & (scores.k == 500)
    j = int(np.argmin(means[1:]))
    scores.loc[mask, ["R", "A0", "integration_mc_se", "dominant_edge"]] = [
        val,
        means[0],
        item["integration_mc_se"],
        str(edges.iloc[j].alpha) + "_" + str(edges.iloc[j].beta),
    ]
    audit.append(dict(patient=pid, R=float(val), cells=int((ids == pid).sum())))
scores.to_csv(OUT / "expanded_graph_scores.csv", index=False)
thresholds = pd.read_csv(OUT / "depth_threshold_sensitivity.csv")
for i, row in thresholds.query(
    'cohort == "D_expanded" and graph == "integrin26" and k == 500'
).iterrows():
    selected = (ids == row.patient) & (data["depth"] >= row.threshold)
    a = data["flows"][selected, ik, :].mean(axis=0)
    thresholds.loc[i, "R"] = a[1:].min() / a[0]
thresholds.to_csv(OUT / "depth_threshold_sensitivity.csv", index=False)


def inversions(a, b):
    upper = np.triu_indices(len(a), 1)
    return int(np.sum(((a[:, None] - a[None, :]) * (b[:, None] - b[None, :]))[upper] < 0))


w = scores.query('graph == "integrin26"').pivot(index="patient", columns="k", values="R")
tw = thresholds.query('cohort == "D_expanded" and graph == "integrin26" and k == 500').pivot(
    index="patient", columns="threshold", values="R"
)
report = dict(
    date="2026-10-19",
    refined_patients=audit,
    R2_to_R500_inversions=inversions(w[2].to_numpy(), w[500].to_numpy()),
    median_R500_minus_R2=float((w[500] - w[2]).median()),
    threshold_inversions={
        str(t): inversions(tw[500].to_numpy(), tw[t].to_numpy()) for t in [1000, 2000]
    },
    extra_integration_seconds=sum(d["seconds"] for d in refined),
)
(ROOT / "integration/refinement_propagation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
