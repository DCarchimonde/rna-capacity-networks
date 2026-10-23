"""Check count provenance, final component summaries and manuscript anchors."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
checks = {}


def check(name, condition):
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


for filename, expected in {
    "development_counts.npz": "15f16a17da11a8b4d3bbc403b15fffd9cc8a51ba2e79a789ed04620be501f044",
    "independent_counts.npz": "25884937355803b84a282ac11510deeee6942726af8b357477d7c551c67a2ee3",
}.items():
    path = ROOT / "data" / filename
    check(filename + "_source_hash", hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    with np.load(path, allow_pickle=False) as z:
        check(filename + "_integer_counts", np.issubdtype(z["counts"].dtype, np.integer))
        check(filename + "_full_depths", np.all(z["counts"].sum(1) <= z["depth"]))
        check(filename + "_depth_eligibility", np.all(z["depth"] >= 500))

base = ROOT / "analyses/review_completion"
with np.load(base / "inputs/atlas_integrins_all_cells.npz") as z:
    check("atlas_cells", len(z["depth"]) == 370115)
    check("atlas_genes", len(z["genes"]) == 26)
    check(
        "atlas_raw_counts",
        np.issubdtype(z["counts"].dtype, np.integer)
        and np.all(z["counts"] >= 0)
        and np.all(z["counts"].sum(1) <= z["depth"]),
    )

with np.load(base / "results/expanded_fullgraph_cellflows.npz") as z:
    check("full_graph_patients", len(z["patients"]) == 62)
    check("full_graph_epithelial_cells", len(z["depth"]) == 113216)
    check("full_graph_24_deletions", z["flows"].shape[-1] == 25)
    scores = pd.read_csv(base / "results/expanded_graph_scores.csv")
    for i, patient in enumerate(z["patients"]):
        means = z["flows"][z["patient_index"] == i].mean(0)
        ratio = means[:, 1:].min(1) / means[:, 0]
        target = (
            scores[(scores.graph == "integrin26") & (scores.patient == patient)]
            .set_index("k")
            .loc[z["ks"]]
        )
        check(
            "component_score_" + str(patient), np.allclose(ratio, target.R, rtol=1e-10, atol=1e-12)
        )

summary = pd.read_csv(ROOT / "analyses/motif_extension/results/competition_summary.csv")
row = summary[(summary.cohort == "D_full26") & (summary.k == 500)].iloc[0]
check("119_rank_reversals", row.inversion_R2_full == 119)
check("43_bivariate_reversals", row.total_inversions_already_in_bivariate == 43)
check("76_shared_capacity_reversals", row.total_inversions_requiring_shared_capacity == 76)
check(
    "median_shared_capacity_fraction",
    abs(row.median_competition_fraction - 0.0527415011370735) < 1e-10,
)

paired = pd.read_csv(
    ROOT / "analyses/second_round/results/paired_permutation_summary.csv", keep_default_na=False
)
expected = {
    ("matched", "alternative", "D"): 16,
    ("matched", "alternative", "R500"): 21,
    ("matched", "null", "D"): 11,
    ("matched", "null", "R500"): 12,
    ("parity", "alternative", "D"): 249,
    ("parity", "alternative", "R500"): 67,
    ("parity", "null", "D"): 11,
    ("parity", "null", "R500"): 11,
}
for (family, setting, statistic), hits in expected.items():
    value = paired[
        (paired.family == family) & (paired.setting == setting) & (paired.statistic == statistic)
    ]
    check(
        "_".join([family, setting, statistic]),
        len(value) == 1
        and int(value.iloc[0].rejections) == hits
        and int(value.iloc[0].repetitions) == 256
        and int(value.iloc[0].undefined_tests) == 0,
    )

mapping = json.loads((ROOT / "source_data_map.json").read_text())
for filename, source in mapping.items():
    a = pd.read_csv(ROOT / "source_data" / filename)
    b = pd.read_csv(ROOT / source)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)
    checks["source_table_" + filename] = True
report = {
    "all_passed": all(checks.values()),
    "checks": len(checks),
    "details": checks,
    "scope": "Checks of supplied counts, component aggregation and reported results; this is not external biological validation.",
}
(ROOT / "review/result_checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Passed {len(checks)} count, component and result checks.")
