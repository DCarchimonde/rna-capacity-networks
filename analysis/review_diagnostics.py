"""Diagnostics of the retained count caches; does not refit or modify source results."""

from pathlib import Path
import json, hashlib, math
from fractions import Fraction as Q
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "review"
OUT.mkdir(exist_ok=True)
rows = []
genes_rows = []
patient_rows = []
occupancy = []
for co, name in [("D", "development_counts.npz"), ("I", "independent_counts.npz")]:
    z = np.load(DATA / name, allow_pickle=False)
    X, L, typ, pi = z["counts"], z["depth"], z["celltype"], z["patient_index"]
    assert np.all(L >= 500) and np.all(X.sum(1) <= L)
    eligible = [
        i
        for i in range(len(z["patients"]))
        if np.sum((pi == i) & (typ == 0)) >= 100 and np.sum((pi == i) & (typ == 1)) >= 30
    ]
    for t, scope in [(0, "Epi"), (1, "Fibro")]:
        mask = typ == t
        if t == 1:
            mask &= np.isin(pi, eligible)
        a, l = X[mask], L[mask]
        ids = pi[mask]
        panel = a[:, [0, 1, 2, 4]].sum(1)
        q = np.quantile(l, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
        row = dict(
            cohort=co,
            scope=scope,
            patients=len(np.unique(ids)),
            cells=len(l),
            depth_mean=float(l.mean()),
            depth_min=q[0],
            depth_p10=q[1],
            depth_p25=q[2],
            depth_median=q[3],
            depth_p75=q[4],
            depth_p90=q[5],
            depth_max=q[6],
            cells_at_500=int(np.sum(l == 500)),
            cells_500_999=int(np.sum(l < 1000)),
            fraction_500_999=float(np.mean(l < 1000)),
            graph_fraction_mean=float(np.mean(panel / l)),
            graph_fraction_median=float(np.median(panel / l)),
            zero_graph_count_fraction=float(np.mean(panel == 0)),
            panel_count_at_least_8_fraction=float(np.mean(panel >= 8)),
        )
        rows.append(row)
        for g, gene in enumerate(z["genes"]):
            present = not (co == "I" and gene == "ITGB3")
            genes_rows.append(
                dict(
                    cohort=co,
                    scope=scope,
                    gene=str(gene),
                    present_in_source=present,
                    mean_observed_fraction=float(np.mean(a[:, g] / l)) if present else None,
                    detection_fraction=float(np.mean(a[:, g] > 0)) if present else None,
                )
            )
        for k in [2, 8, 32, 128, 500]:
            tail = hypergeom.sf(2, l, panel, k)
            assert np.all(np.isfinite(tail)) and np.all((tail >= 0) & (tail <= 1))
            occupancy.append(
                dict(
                    cohort=co,
                    scope=scope,
                    k=k,
                    mean_expected_panel_count=float(np.mean(k * panel / l)),
                    mean_conditional_probability_panel_count_ge3=float(tail.mean()),
                    median_conditional_probability_panel_count_ge3=float(np.median(tail)),
                    probability_ge3_p95=float(np.quantile(tail, 0.95)),
                )
            )
        for i in np.unique(ids):
            m = ids == i
            patient_rows.append(
                dict(
                    cohort=co,
                    scope=scope,
                    patient=str(z["patients"][i]),
                    cells=int(m.sum()),
                    median_depth=float(np.median(l[m])),
                    mean_graph_fraction=float(np.mean(panel[m] / l[m])),
                    fraction_500_999=float(np.mean(l[m] < 1000)),
                    graph_detection_fraction=float(np.mean(panel[m] > 0)),
                )
            )
pd.DataFrame(rows).to_csv(OUT / "retained_depth_and_sparsity.csv", index=False)
pd.DataFrame(genes_rows).to_csv(OUT / "gene_detection_and_fraction.csv", index=False)
pd.DataFrame(patient_rows).to_csv(OUT / "patient_depth_and_sparsity.csv", index=False)
pd.DataFrame(occupancy).to_csv(OUT / "conditional_graph_occupancy.csv", index=False)

# On each split, both methods use the same patients whose native graph-panel fit
# converged in BOTH halves. Their raw H is used without projection or fallback.
native_rows = []
split_rows = []
for co in ["D", "I"]:
    z = np.load(DATA / f"{co}_H_scores.npz", allow_pickle=False)
    fit = pd.read_csv(DATA / f"{co}_fit_status.csv.gz")
    patient = z["patients"]
    n = len(patient)
    i, j = np.triu_indices(n, 1)
    edge = list(z["edges"]).index("ITGAV_ITGB6")
    method_indices = [
        list(z["methods"]).index(x) for x in ["equal_cell_moment", "CSCORE_graph_raw"]
    ]
    half = z["halves"][:, :, 0, :, edge]
    finite = np.all(np.isfinite(half[:, :, method_indices]), axis=2)
    ok = np.zeros((40, n), dtype=bool)
    f = fit[(fit.kind == "cell_half") & (fit.cell_type == "Epi") & (fit.variant == "graph")]
    assert len(f) == 40 * n
    for r in f.itertuples():
        ok[int(r.role), list(patient).index(r.patient)] = r.actual_converged == True
    ok &= finite
    valid = []
    outcomes = [[], []]
    for r in range(0, 40, 2):
        common = ok[r] & ok[r + 1] & z["Epi_eligible"]
        v = common[i] & common[j]
        valid.append(v)
        splitrow = dict(
            cohort=co,
            split=r // 2,
            converged_patients=int(common.sum()),
            common_pairs=int(v.sum()),
            total_pairs=len(i),
        )
        assert v.sum() > 0
        for m, mi in enumerate(method_indices):
            signs = []
            for role in [r, r + 1]:
                a = half[role, :, mi]
                diff = a[i] - a[j]
                tol = 64 * np.finfo(float).eps * np.maximum(np.abs(a[i]), np.abs(a[j]))
                signs.append(np.where(np.isfinite(diff) & (np.abs(diff) > tol), np.sign(diff), 0))
            success = signs[0] * signs[1] > 0
            outcomes[m].append(success)
            splitrow[["equal_cell_retention", "native_retention"][m]] = float(success[v].mean())
        split_rows.append(splitrow)
    valid = np.array(valid)
    outcomes = np.array(outcomes)
    point = np.array([(o & valid).sum(axis=1) / valid.sum(axis=1) for o in outcomes]).mean(axis=1)
    rng = np.random.default_rng(261018610 + (co == "I"))
    count = rng.multinomial(n, np.full(n, 1 / n), size=2000)
    weights = count[:, i] * count[:, j]
    denominator = weights @ valid.T
    distributions = []
    for o in outcomes:
        numerator = weights @ (o & valid).T
        ratio = np.divide(
            numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0
        )
        distributions.append(np.nanmean(ratio, axis=1))
    distributions = np.array(distributions)
    assert np.all(np.isfinite(distributions))
    for m, method in enumerate(["equal_cell_moment", "CSCORE_graph_raw"]):
        lo, hi = np.quantile(distributions[m], [0.025, 0.975])
        native_rows.append(
            dict(
                cohort=co,
                method=method,
                retention=point[m],
                patient_95_low=lo,
                patient_95_high=hi,
                original_patients=n,
                converged_patients_min=min(
                    x["converged_patients"] for x in split_rows if x["cohort"] == co
                ),
                converged_patients_max=max(
                    x["converged_patients"] for x in split_rows if x["cohort"] == co
                ),
                mean_pair_coverage=float(valid.mean()),
                paired_difference_equal_minus_native=float(point[0] - point[1]),
                paired_95_low=float(np.quantile(distributions[0] - distributions[1], 0.025)),
                paired_95_high=float(np.quantile(distributions[0] - distributions[1], 0.975)),
                bootstrap_repetitions=2000,
                bootstrap_repetitions_with_any_empty_split=int(
                    np.any(denominator == 0, axis=1).sum()
                ),
            )
        )
pd.DataFrame(native_rows).to_csv(OUT / "native_CSCORE_common_converged.csv", index=False)
pd.DataFrame(split_rows).to_csv(OUT / "native_CSCORE_split_coverage.csv", index=False)

# Resolution transition computed using the existing
# strict nonzero direction rule and the same qualified cells.
spectra = pd.read_csv(DATA / "patient_resolution_spectra.csv")
transition = []
for (co, scope), d in spectra[spectra.graph == "common3"].groupby(["cohort", "scope"]):
    p = d.pivot(index="patient", columns="k", values="R_k")
    i, j = np.triu_indices(len(p), 1)
    a = p[8].to_numpy()
    b = p[500].to_numpy()
    x = a[i] - a[j]
    y = b[i] - b[j]
    transition.append(
        dict(
            cohort=co,
            scope=scope,
            patients=len(p),
            pairs=len(i),
            inversions=int(np.sum(x * y < 0)),
            unresolved=int(np.sum((x == 0) | (y == 0))),
        )
    )
pd.DataFrame(transition).to_csv(OUT / "R8_to_R500_inversions.csv", index=False)


# Exact counterexamples: the ratio's bias need not have the numerator's sign.
def flows(p):
    a, v, b, c, d = p
    return [
        min(a + v, v + b, b + c + d),
        min(v, b + c + d),
        min(a, b) + min(v, c + d),
        min(a + v, v + b, b + d),
        min(a + v, v + b, b + c),
    ]


def ratio_example(states):
    f = [flows(p) for p in states]
    n = len(f)
    means = [sum(x[j] for x in f) / n for j in range(5)]
    target = min(means[1:]) / means[0]
    expected = sum(min(x[1:]) / x[0] for x in f) / n
    return {
        "population_ratio": str(target),
        "expected_one_cell_ratio": str(expected),
        "bias": str(expected - target),
    }


up = ratio_example([[Q(9, 20), Q(1, 20), Q(9, 20), Q(1, 20), Q(0)], [Q(1, 20)] * 4 + [Q(0)]])
down = ratio_example(
    [[Q(9, 20), Q(1, 20), Q(9, 20), Q(1, 20), Q(0)], [Q(1, 20), Q(9, 20), Q(1, 20), Q(9, 20), Q(0)]]
)
assert Q(up["bias"]) > 0 and Q(down["bias"]) < 0
report = {
    "all_checks_passed": True,
    "sampling_target": "Previously retained cells only; no claim about removed cells or unobserved absolute RNA.",
    "occupancy_interpretation": "Conditional hypergeometric probability in the retained count sample. At least three graph molecules is necessary but not sufficient for a three-distinct-gene competition pattern; this is not a statistical power calculation.",
    "native_comparison": "Raw native output; strict actual convergence in both halves; same eligible patients and pair denominator for both methods, reported per split. Selection changes the population, so full-cohort results remain in the prior tables.",
    "ratio_bias_counterexamples_cell_local_limit": {"positive": up, "negative": down},
    "comb_500_500": math.comb(500, 500),
    "comb_501_500": math.comb(501, 500),
    "comb_510_500": math.comb(510, 500),
    "input_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(DATA.glob("*"))
        if p.is_file()
    },
}
(OUT / "review_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    pd.DataFrame(rows)[
        ["cohort", "scope", "cells", "depth_median", "fraction_500_999", "graph_fraction_mean"]
    ].to_string(index=False)
)
print(pd.DataFrame(occupancy).query("k in [8,500]").to_string(index=False))
print(pd.DataFrame(native_rows).to_string(index=False))
print(pd.DataFrame(transition).to_string(index=False))
