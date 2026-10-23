"""Post-primary descriptive decomposition using existing author subtype labels."""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(root):
    root = Path(root)
    primary = json.loads((root / "results/independent/run.json").read_text())
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    m = pd.read_csv(
        root / "inputs/analysis_metadata.tsv.gz",
        sep="\t",
        usecols=["cell_id", "minor_cell_type"],
        dtype=str,
    ).set_index("cell_id")
    z = np.load(root / "inputs/prepared/qualified_counts.npz", allow_pickle=False)
    records, group_records = [], []
    for i, patient in enumerate(z["patients"]):
        sel = (z["patient_index"] == i) & (z["celltype"] == 0)
        l = z["depth"][sel].astype(float)
        a, b = z["counts"][sel, 1] / l, z["counts"][sel, 4] / l
        subtype = m.loc[z["cell_id"][sel], "minor_cell_type"].to_numpy()
        assert not pd.isna(subtype).any()
        n = len(l)
        h = np.mean(a * b * l / (l - 1))
        naive = np.mean(a * b)
        u = (a.mean() * b.mean() * n - naive) / (n - 1)
        within, group_product = 0.0, 0.0
        for name in pd.unique(subtype):
            ix = subtype == name
            weight = ix.mean()
            ma, mb, product = a[ix].mean(), b[ix].mean(), (a[ix] * b[ix]).mean()
            within += weight * (product - ma * mb)
            group_product += weight * ma * mb
            group_records.append(
                {
                    "patient": patient,
                    "author_Epi_subtype": name,
                    "cells": int(ix.sum()),
                    "cell_weight": weight,
                    "mean_ITGAV_fraction": ma,
                    "mean_ITGB6_fraction": mb,
                    "mean_fraction_product": product,
                }
            )
        between = group_product - a.mean() * b.mean()
        # This is an exact decomposition of the finite-data estimate, not a
        # separately unbiased biological within-subtype covariance estimator.
        parts = [n / (n - 1) * within, n / (n - 1) * between, h - naive]
        records.append(
            {
                "patient": patient,
                "cells": n,
                "author_subtypes": len(pd.unique(subtype)),
                "observed_within_subtype_component": parts[0],
                "observed_between_subtype_component": parts[1],
                "factorial_molecule_correction": parts[2],
                "H_minus_U": h - u,
                "decomposition_abs_error": abs(sum(parts) - (h - u)),
            }
        )
    df = pd.DataFrame(records)
    assert df.decomposition_abs_error.max() < 1e-20
    df.to_csv(root / "results/epithelial_subtype_decomposition.csv", index=False)
    pd.DataFrame(group_records).to_csv(root / "results/epithelial_subtype_means.csv", index=False)
    components = [
        "observed_within_subtype_component",
        "observed_between_subtype_component",
        "factorial_molecule_correction",
        "H_minus_U",
    ]
    result = {
        "status": "POST_PRIMARY_DESCRIPTIVE_ANALYSIS",
        "started_utc": started,
        "primary_finished_utc": primary["finished_utc"],
        "primary_specification_changed": False,
        "grouping": "Existing author minor_cell_type; all epithelial cells retained; no new subtype size cutoffs",
        "patient_mean_components": {c: float(df[c].mean()) for c in components},
        "max_decomposition_abs_error": float(df.decomposition_abs_error.max()),
        "interpretation": "Exact decomposition of observed RNA-proportion H-U; no regulatory or latent subtype covariance claim",
    }
    (root / "results/epithelial_subtype_decomposition.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
