"""Summarize fixed outputs and draw standalone scientific figures."""

import argparse
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from estimate_h import pair_metrics
from prepare_counts import sha


def main(root, development):
    root, development = Path(root), Path(development)
    r = root / "results/independent"
    edge = "ITGAV_ITGB6"
    coupling = pd.read_csv(r / "coupling_summary.csv")
    split = pd.read_csv(r / "split_summary.csv")
    umi = pd.read_csv(r / "UMI_roles.csv")
    table = pd.read_csv(r / "patient_H.csv")
    p = table.loc[(table.scope == "Epi") & (table.edge == edge)].copy()
    q = pd.read_csv(root / "inputs/prepared/qualification.csv")
    with np.load(root / "inputs/prepared/qualified_counts.npz", allow_pickle=False) as z:
        depth, x = z["depth"], z["counts"]
        pi, typ, patients = z["patient_index"], z["celltype"], z["patients"]
        presence = z["gene_present"]
    with np.load(r / "scores.npz", allow_pickle=False) as z:
        masks_correct = all(np.isnan(z[k][..., 2]).all() for k in ["full", "halves", "thin"])
        full = z["full"]
        halves = z["halves"]
    real_checks = {
        "twelve_selected_specimens": len(q) == 12 and q["sample"].is_unique,
        "all_matrix_cells_match_author_annotation": q.metadata_unmatched_cells.eq(0).all()
        and q.annotated_cells_missing_from_matrix.eq(0).all(),
        "raw_count_depth_threshold": bool((depth >= 500).all()),
        "count_cache_primary_gene_rows_present": bool(presence[:, [1, 4]].all()),
        "ITGB3_missing_in_every_specimen": bool((~presence[:, 3]).all()),
        "missing_edge_is_NA_in_every_saved_role": bool(masks_correct),
    }
    errors = []
    codetection = []
    meta = pd.read_csv(
        root / "inputs/analysis_metadata.tsv.gz",
        sep="\t",
        usecols=["cell_id", "nCount_spliced"],
        dtype={"cell_id": str},
    ).set_index("cell_id")
    with np.load(root / "inputs/prepared/qualified_counts.npz", allow_pickle=False) as z:
        real_checks["all_cached_denominators_equal_author_spliced_totals"] = bool(
            np.array_equal(depth, meta.loc[z["cell_id"], "nCount_spliced"].to_numpy())
        )
    for i, patient in enumerate(patients):
        for t in [0, 1]:
            keep = (pi == i) & (typ == t)
            a, b, l = x[keep, 1].astype(float), x[keep, 4].astype(float), depth[keep].astype(float)
            n = len(l)
            copos = int(((a > 0) & (b > 0)).sum())
            left, right = n // 2, n - n // 2

            def all_on_one_side(size):
                return (
                    Fraction(math.comb(size, copos), math.comb(n, copos))
                    if size >= copos
                    else Fraction(0)
                )

            both = float(1 - all_on_one_side(left) - all_on_one_side(right)) if copos else 0.0
            qr = q.loc[q.patient == patient].iloc[0]
            codetection.append(
                {
                    "patient": patient,
                    "scope": ["Epi", "Fibro"][t],
                    "cells": n,
                    "ITGAV_positive": int((a > 0).sum()),
                    "ITGB6_positive": int((b > 0).sum()),
                    "codetected_cells": copos,
                    "scope_eligible": bool(qr.Epi_eligible if t == 0 else qr.two_type_eligible),
                    "conditional_probability_nonzero_in_both_halves": both,
                    "observed_nonzero_both_halves_fraction": float(
                        ((halves[::2, i, t, 0, 3] > 0) & (halves[1::2, i, t, 0, 3] > 0)).mean()
                    ),
                }
            )
            if n < 2:
                continue
            # Independent scalar expressions for the same fixed estimands.
            direct_h = np.sum(a * b / (l * (l - 1))) / n
            ya, yb = a / l, b / l
            direct_u = (ya.mean() * yb.mean() * n - (ya * yb).mean()) / (n - 1)
            errors.extend([abs(direct_h - full[i, t, 0, 3]), abs(direct_u - full[i, t, 2, 3])])
    real_checks["direct_primary_H_and_U_agree"] = max(errors) < 1e-20
    missing = split.loc[split.edge == "ITGAV_ITGB3"]
    real_checks["missing_edge_summary_is_NA"] = (
        missing.patients.eq(0).all() and missing.reference_resolution.isna().all()
    )
    assert all(real_checks.values()), real_checks
    (root / "results/real_run_verification.json").write_text(
        json.dumps(
            {
                "all_passed": True,
                "checks": {k: bool(v) for k, v in real_checks.items()},
                "max_primary_H_U_abs_difference": max(errors),
                "analysis_sha256": sha(root / "scripts/analyze_h.py"),
            },
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(codetection).to_csv(root / "results/codetection_counts.csv", index=False)

    records = []
    for label, path in [("Development", development), ("Independent", r)]:
        c, s, u = [
            pd.read_csv(path / name)
            for name in ["coupling_summary.csv", "split_summary.csv", "UMI_roles.csv"]
        ]
        for scope in ["Epi", "Fibro", "balanced"]:
            a = c.loc[(c.scope == scope) & (c.edge == edge)].iloc[0]
            b = s.loc[(s.scope == scope) & (s.edge == edge) & (s.method == "cell_factorial")].iloc[
                0
            ]
            v = u.loc[(u.scope == scope) & (u.edge == edge) & (u.method == "cell_factorial")]
            records.append(
                {
                    "cohort": label,
                    "scope": scope,
                    "patients": int(a.patients),
                    "mean_H": a.mean_H,
                    "mean_U": a.mean_U,
                    "H_U_ratio_of_means": a.H_to_U_ratio_of_means,
                    "H_minus_U": a.mean_H_minus_U,
                    "coupling_95_low": a.patient_95_low,
                    "coupling_95_high": a.patient_95_high,
                    "positive_patients": int(a.positive_patients),
                    "split_retention": b.all_pair_retention,
                    "split_resolution": b.reference_resolution,
                    "split_95_low": b.all_pair_retention_patient_95_low,
                    "split_95_high": b.all_pair_retention_patient_95_high,
                    "UMI_retention": v.all_pair_retention.mean(),
                }
            )
    comparison = pd.DataFrame(records)
    comparison.to_csv(root / "results/cohort_comparison.csv", index=False)
    reference = root / "inputs/development_reference"
    reference.mkdir(exist_ok=True)
    for name in [
        "coupling_summary.csv",
        "split_summary.csv",
        "UMI_roles.csv",
        "patient_H.csv",
        "run.json",
    ]:
        if (development / name).resolve() != (reference / name).resolve():
            shutil.copy2(development / name, reference / name)
    pm = pair_metrics(p.H.to_numpy(), p.crosscell_U.to_numpy())
    summary = {
        "cohort_comparison": records,
        "independent_primary_patient_order_summary": {
            "distinct_patient_pairs": pm["distinct_pairs"],
            "same_full_data_direction_pairs": round(
                pm["all_pair_retention"] * pm["distinct_pairs"]
            ),
            "opposite_full_data_direction_pairs": round(
                (1 - pm["all_pair_retention"]) * pm["distinct_pairs"]
            ),
            "D_H_vs_U": pm["D"],
            "interpretation": "Different RNA targets; disagreement is not an accuracy estimate",
        },
        "selected_matrix_cells": int(q.matrix_cells.sum()),
        "cached_Epi_cells": int((typ == 0).sum()),
        "cached_Fibro_cells": int((typ == 1).sum()),
        "biological_patients": len(patients),
        "count_layer": "all-gene totals match author nCount_spliced and RNA alias",
    }
    (root / "results/replication_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--development-results", required=True)
    a = p.parse_args()
    main(a.root, a.development_results)
