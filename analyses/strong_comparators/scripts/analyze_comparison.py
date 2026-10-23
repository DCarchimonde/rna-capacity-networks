"""Paired comparisons, retaining all patient pairs and all official fit failures."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from cscore_adapter import METHODS, r2_from_h
from numeric_metrics import pair_metrics, difference

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/comparison_spec.json").read_text())
BASELINES = ["samecell_fraction", "CSCORE_graph_fallback", "CSCORE_pair_fallback"]


def simulation():
    d = pd.read_csv(
        ROOT / "results/H_simulation_metrics.csv.gz", keep_default_na=False, na_values=[""]
    )
    endpoints = [
        "strict_accuracy",
        "RMSE",
        "MAE",
        "mean_bias",
        "point_coverage",
        "reference_resolution",
        "query_resolution",
    ]
    d.groupby(
        ["source", "family", "depth_design", "scope", "target", "target_kind", "method"],
        dropna=False,
    )[endpoints].mean().reset_index().to_csv(ROOT / "results/H_simulation_summary.csv", index=False)
    primary = d[(d.scope == "balanced") & (d.target == "ITGAV_ITGB6")]
    comparisons = []
    for keys, g in primary.groupby(["source", "family", "depth_design"], sort=False):
        for comparator in BASELINES:
            for metric in ["strict_accuracy", "RMSE"]:
                strata = []
                for _, case in g.groupby("case"):
                    a = case[case.method == "equal_cell_moment"].set_index("replicate")[metric]
                    b = case[case.method == comparator].set_index("replicate")[metric]
                    delta = (a - b).dropna().to_numpy()
                    if len(delta):
                        strata.append(delta)
                if not strata:
                    continue
                rng = np.random.default_rng(2610109292)
                boots = np.mean(
                    [v[rng.integers(len(v), size=(5000, len(v)))].mean(1) for v in strata], axis=0
                )
                lo, hi = np.quantile(boots, [0.025, 0.975])
                diff = float(np.mean([v.mean() for v in strata]))
                comparisons.append(
                    {
                        "source": keys[0],
                        "family": keys[1],
                        "depth_design": keys[2],
                        "comparator": comparator,
                        "endpoint": metric,
                        "equal_cell_minus_comparator": diff,
                        "MC_95_low": lo,
                        "MC_95_high": hi,
                        "cohorts": sum(map(len, strata)),
                        "equally_weighted_cases": len(strata),
                    }
                )
    pd.DataFrame(comparisons).to_csv(
        ROOT / "results/H_simulation_paired_comparison.csv", index=False
    )
    fits = pd.read_csv(
        ROOT / "results/H_simulation_fit_status.csv.gz",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    fits["exception"] = fits.fit_status == "exception"
    fits["returned"] = fits.fit_status == "returned"
    fits["not_needed"] = fits.fit_status == "not_needed_zero_edges"
    fits["nonconverged_return"] = fits.returned & ~(
        fits.actual_converged.fillna(False).astype(bool)
    )
    cols = [
        "exception",
        "returned",
        "not_needed",
        "nonconverged_return",
        "fallback_edges",
        "negative_H_edges",
    ]
    fits.groupby(["source", "family", "depth_design", "variant"])[cols].agg(["sum", "mean"]).to_csv(
        ROOT / "results/H_simulation_fit_summary.csv"
    )


def bootstrap_retention(reference, query, bt):
    """Paired patient resamples; duplicated identities do not form patient pairs."""
    n = reference.shape[1]
    i, j = np.triu_indices(n, 1)
    distinct = bt[:, i] != bt[:, j]
    den = distinct.sum(1)
    valid = den > 0
    values = np.zeros(len(bt))
    resolution = np.zeros(len(bt))
    for a, b in zip(reference, query):
        aa, bb = a[bt], b[bt]
        da, db = difference(aa[:, i], aa[:, j]), difference(bb[:, i], bb[:, j])
        resolved = distinct & np.isfinite(da) & (da != 0)
        retained = resolved & np.isfinite(db) & (da * db > 0)
        values += np.divide(retained.sum(1), den, out=np.zeros(len(bt)), where=valid)
        resolution += np.divide(resolved.sum(1), den, out=np.zeros(len(bt)), where=valid)
    return values / len(reference), resolution / len(reference), valid


def real():
    summaries = []
    gains = []
    r2_agreements = []
    fits = []
    for name in ["D", "I"]:
        z = np.load(ROOT / f"results/{name}_H_scores.npz")
        full, half, thin = z["full"], z["halves"], z["thin"]
        eligibility = [
            np.flatnonzero(z["Epi_eligible"]),
            np.flatnonzero(z["dual_eligible"]),
            np.flatnonzero(z["dual_eligible"]),
        ]
        targets = [("H_ITGAV_ITGB6", full[:, :, :, 3], half[:, :, :, :, 3], thin[:, :, :, :, 3])]
        for common in ([False, True] if name == "D" else [True]):
            label = "common3" if common else "full4"
            targets.append(
                (
                    "R2_" + label,
                    r2_from_h(full, common),
                    r2_from_h(half, common),
                    r2_from_h(thin, common),
                )
            )
        for target, f, h, u in targets:
            for typ, scope in enumerate(["Epi", "Fibro", "balanced"]):
                ix = eligibility[typ]
                n = len(ix)
                rng = np.random.default_rng(2610109291)
                bt = rng.integers(n, size=(2000, n))
                all_boot = {}
                for mi, method in enumerate(METHODS):
                    if target.startswith("R2") and mi == 5:
                        continue
                    for kind, a, b in [
                        ("cell_split", h[:, ix, typ, mi], h[np.arange(len(h)) ^ 1][:, ix, typ, mi]),
                        ("UMI_thin", np.tile(f[ix, typ, mi], (len(u), 1)), u[:, ix, typ, mi]),
                    ]:
                        metrics = [pair_metrics(aa, bb) for aa, bb in zip(a, b)]
                        if method in ["equal_cell_moment", *BASELINES]:
                            boot, bres, valid = bootstrap_retention(a, b, bt)
                            lo, hi = np.quantile(boot[valid], [0.025, 0.975])
                        else:
                            boot = np.full(len(bt), np.nan)
                            valid = np.ones(len(bt), bool)
                            lo = hi = np.nan
                        all_boot[(method, kind)] = (boot, valid)
                        summaries.append(
                            {
                                "cohort": name,
                                "scope": scope,
                                "target": target,
                                "method": method,
                                "kind": kind,
                                "patients": n,
                                "all_pair_retention": np.mean(
                                    [v["all_pair_retention"] for v in metrics]
                                ),
                                "reference_resolution": np.mean(
                                    [v["reference_resolution"] for v in metrics]
                                ),
                                "mean_finite_patients": np.mean(
                                    [v["finite_patients"] for v in metrics]
                                ),
                                "patient_95_low": lo,
                                "patient_95_high": hi,
                                "full_mean_score": float(np.nanmean(f[ix, typ, mi])),
                            }
                        )
                for comparator in BASELINES:
                    if target.startswith("R2") and comparator == "CSCORE_pair_fallback":
                        continue
                    for kind in ["cell_split", "UMI_thin"]:
                        ba, va = all_boot[("equal_cell_moment", kind)]
                        bb, vb = all_boot[(comparator, kind)]
                        valid = va & vb
                        lo, hi = np.quantile((ba - bb)[valid], [0.025, 0.975])
                        sa = next(
                            row
                            for row in reversed(summaries)
                            if row["cohort"] == name
                            and row["scope"] == scope
                            and row["target"] == target
                            and row["method"] == "equal_cell_moment"
                            and row["kind"] == kind
                        )
                        sb = next(
                            row
                            for row in reversed(summaries)
                            if row["cohort"] == name
                            and row["scope"] == scope
                            and row["target"] == target
                            and row["method"] == comparator
                            and row["kind"] == kind
                        )
                        gains.append(
                            {
                                "cohort": name,
                                "scope": scope,
                                "target": target,
                                "kind": kind,
                                "comparator": comparator,
                                "patients": n,
                                "equal_cell_minus_comparator": sa["all_pair_retention"]
                                - sb["all_pair_retention"],
                                "patient_95_low": lo,
                                "patient_95_high": hi,
                            }
                        )
                if target.startswith("R2"):
                    for mi, method in enumerate(METHODS[:-1]):
                        a = f[ix, typ, 0]
                        b = f[ix, typ, mi]
                        i, j = np.triu_indices(n, 1)
                        da, db = difference(a[i], a[j]), difference(b[i], b[j])
                        finite = np.isfinite(da) & np.isfinite(db)
                        r2_agreements.append(
                            {
                                "cohort": name,
                                "scope": scope,
                                "target": target,
                                "method": method,
                                "all_pairs": len(i),
                                "opposite_pairs": int((finite & (da * db < 0)).sum()),
                                "same_pairs": int((finite & (da * db > 0)).sum()),
                                "unresolved_pairs": int((~finite | (da == 0) | (db == 0)).sum()),
                                "max_abs_score_difference": float(np.nanmax(abs(a - b))),
                            }
                        )
        status = pd.read_csv(ROOT / f"results/{name}_fit_status.csv.gz")
        status["cohort"] = name
        fits.append(status)
    pd.DataFrame(summaries).to_csv(ROOT / "results/real_primary_summary.csv", index=False)
    pd.DataFrame(gains).to_csv(ROOT / "results/real_primary_paired_comparison.csv", index=False)
    pd.DataFrame(r2_agreements).to_csv(ROOT / "results/real_R2_agreement.csv", index=False)
    fit = pd.concat(fits, ignore_index=True)
    fit["exception"] = fit.fit_status == "exception"
    fit["returned"] = fit.fit_status == "returned"
    fit["not_needed"] = fit.fit_status == "not_needed_zero_edges"
    fit["nonconverged_return"] = fit.returned & ~fit.actual_converged.fillna(False).astype(bool)
    fit.groupby(["cohort", "kind", "cell_type", "variant"])[
        [
            "exception",
            "returned",
            "not_needed",
            "nonconverged_return",
            "fallback_edges",
            "negative_H_edges",
        ]
    ].agg(["sum", "mean"]).to_csv(ROOT / "results/real_fit_summary.csv")


if __name__ == "__main__":
    simulation()
    real()
    print("Paired simulation and patient comparison tables saved", flush=True)
