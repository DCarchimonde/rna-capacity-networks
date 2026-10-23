from pathlib import Path
import json, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT.parents[1]
O = ROOT / "results"
out = {}
z = np.load(ROOT / "inputs/atlas_integrins_all_cells.npz")
m = (z["tissue"] == "T") & (z["celltype"] == "Epi")
out["D_source_epithelial"] = {
    "cells": int(m.sum()),
    "minimum_depth": int(z["depth"][m].min()),
    "below_500": int(np.sum(z["depth"][m] < 500)),
}
mi = pd.read_csv(
    P / "analyses/independent_pair/inputs/analysis_metadata.tsv.gz", sep="\t", low_memory=False
)
zi = np.load(P / "data/independent_counts.npz")
selected = np.unique(zi["sample_id"])
m = mi.sample_id.isin(selected) & (mi.main_cell_type == "Epithelial")
out["I_source_epithelial"] = {
    "cells": int(m.sum()),
    "minimum_depth": int(mi.loc[m, "nCount_spliced"].min()),
    "below_500": int(np.sum(mi.loc[m, "nCount_spliced"] < 500)),
}
score = pd.read_csv(O / "expanded_graph_scores.csv")
graphs = []
for graph, d in score.groupby("graph"):
    w = d.pivot(index="patient", columns="k", values="R")
    i, j = np.triu_indices(len(w), 1)
    a = w[2].to_numpy()
    b = w[500].to_numpy()
    graphs.append(
        dict(
            graph=graph,
            patients=len(w),
            pairs=len(i),
            R2_R500_inversions=int(np.sum((a[i] - a[j]) * (b[i] - b[j]) < 0)),
            R2_min=float(a.min()),
            R2_max=float(a.max()),
            R500_min=float(b.min()),
            R500_max=float(b.max()),
            median_R500_minus_R2=float(np.median(b - a)),
            maximum_absolute_difference=float(np.max(abs(a - b))),
        )
    )
out["expanded_graphs"] = graphs
depth = pd.read_csv(O / "depth_threshold_sensitivity.csv")
rr = []
for (c, g), v in depth.query("k==500").groupby(["cohort", "graph"]):
    w = v.pivot(index="patient", columns="threshold", values="R")
    a = w[500].to_numpy()
    i, j = np.triu_indices(len(a), 1)
    for t in [1000, 2000]:
        b = w[t].to_numpy()
        rr.append(
            dict(
                cohort=c,
                graph=g,
                threshold=t,
                inversions=int(np.sum((a[i] - a[j]) * (b[i] - b[j]) < 0)),
                pairs=len(i),
                maximum_absolute_score_change=float(np.max(abs(a - b))),
                mean_patient_retained_fraction=float(v[v.threshold == t].retained_fraction.mean()),
            )
        )
out["depth_threshold_changes"] = rr
panel = pd.read_csv(O / "panel_conditional_selection.csv")
out["panel_coverage"] = [
    dict(
        cohort=c,
        n=int(n),
        retained_cells=int(g.eligible_cells.sum()),
        total_cells=int(g.total_cells.sum()),
        pooled_cell_coverage=float(g.eligible_cells.sum() / g.total_cells.sum()),
        median_patient_coverage=float(g.coverage.median()),
        zero_eligible_patients=int((g.eligible_cells == 0).sum()),
    )
    for (c, n), g in panel.groupby(["cohort", "panel_resolution"])
]
out["checks"] = {
    "every_score_finite": bool(np.isfinite(score.R).all()),
    "every_score_in_unit_interval": bool(score.R.between(0, 1).all()),
    "every_source_matched_original_counts": json.loads(
        (O / "real_expansion_checks.json").read_text()
    )["original_23_patient_projection_exact"],
}
(O / "revision_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
