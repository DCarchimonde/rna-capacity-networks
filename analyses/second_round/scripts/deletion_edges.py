"""Recompute worst deletions from the current, refined cell-level components."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
source = PROJECT / "analyses/review_completion/results/expanded_fullgraph_cellflows.npz"
z = np.load(source)
edges = pd.read_csv(
    PROJECT / "analyses/review_completion/inputs/integrin_edges_with_provenance.csv"
)
names = (edges.alpha + "_" + edges.beta).tolist()
scores = (
    pd.read_csv(PROJECT / "analyses/review_completion/results/expanded_graph_scores.csv")
    .query('graph == "integrin26"')
    .set_index(["patient", "k"])
)
rows, all_edges = [], []
for ip, patient in enumerate(z["patients"]):
    cell = z["patient_index"] == ip
    means = z["flows"][cell].mean(axis=0)
    for ik, k in enumerate(z["ks"]):
        a = means[ik]
        order = np.argsort(a[1:], kind="stable")
        j, second = order[:2]
        tol = 64 * np.finfo(float).eps * max(abs(a[j + 1]), abs(a[second + 1]))
        ties = np.flatnonzero(a[1:] <= a[j + 1] + tol)
        old = scores.loc[(patient, k)]
        assert abs(old.R - a[j + 1] / a[0]) < 1e-12
        assert old.dominant_edge == names[j]
        rows.append(
            dict(
                patient=patient,
                k=int(k),
                cells=int(cell.sum()),
                A0=a[0],
                R=a[j + 1] / a[0],
                worst_edge=names[j],
                second_edge=names[second],
                tied_worst_edges=";".join(names[t] for t in ties),
                number_tied=len(ties),
                normalized_margin=(a[second + 1] - a[j + 1]) / a[0],
            )
        )
        for e, name in enumerate(names):
            all_edges.append(
                dict(
                    patient=patient,
                    k=int(k),
                    edge=name,
                    deletion_flow=a[e + 1],
                    deletion_loss=a[0] - a[e + 1],
                    normalized_loss=(a[0] - a[e + 1]) / a[0],
                )
            )
df = pd.DataFrame(rows)
df.to_csv(OUT / "worst_deletion_by_patient.csv", index=False)
pd.DataFrame(all_edges).to_csv(OUT / "all_deletion_components.csv", index=False)
counts = df.groupby(["k", "worst_edge"]).size().rename("patients").reset_index()
counts.to_csv(OUT / "worst_deletion_counts.csv", index=False)
w = df.pivot(index="patient", columns="k", values="worst_edge")
switch = w.loc[w[2] != w[500]].reset_index().rename(columns={k: f"edge_k{k}" for k in w.columns})
switch.to_csv(OUT / "worst_deletion_switches.csv", index=False)
report = dict(
    patient_count=len(w),
    switches_2_to_500=len(switch),
    switch_patients=switch.patient.tolist(),
    exact_or_numerical_ties=int((df.number_tied > 1).sum()),
    counts=counts.to_dict("records"),
    switching_patient_details=df.query("patient in @switch.patient and k in [2,500]").to_dict(
        "records"
    ),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    interpretation="conditional on the observed cells and stored integration; descriptive edge identity, not a test of biological function",
)
(OUT / "worst_deletion_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
