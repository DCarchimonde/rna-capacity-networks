"""Refresh manuscript-facing tables from their canonical analysis outputs."""

import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
mapping = json.loads((ROOT / "source_data_map.json").read_text())
for name, source in mapping.items():
    shutil.copyfile(ROOT / source, ROOT / "source_data" / name)

data_mapping = {
    "D_H_scores.npz": "analyses/strong_comparators/results/D_H_scores.npz",
    "I_H_scores.npz": "analyses/strong_comparators/results/I_H_scores.npz",
    "D_fit_status.csv.gz": "analyses/strong_comparators/results/D_fit_status.csv.gz",
    "I_fit_status.csv.gz": "analyses/strong_comparators/results/I_fit_status.csv.gz",
    "H_simulation_paired_comparison.csv": "analyses/strong_comparators/results/H_simulation_paired_comparison.csv",
    "R8_simulation_summary.csv": "analyses/strong_comparators/results/R8_simulation_summary.csv",
    "patient_resolution_spectra.csv": "analyses/molecular_resolution/results/patient_spectrum.csv",
    "resolution_cohort_summary.csv": "analyses/molecular_resolution/results/cohort_summary.csv",
    "resolution_transitions.csv": "analyses/molecular_resolution/results/resolution_transitions.csv",
}
for name, source in data_mapping.items():
    shutil.copyfile(ROOT / source, ROOT / "data" / name)

rows = []
for cohort, filename in [("D", "development_counts.npz"), ("I", "independent_counts.npz")]:
    with np.load(ROOT / "data" / filename) as data:
        for index, patient in enumerate(data["patients"]):
            selected = data["patient_index"] == index
            epithelial = int(np.sum(selected & (data["celltype"] == 0)))
            fibroblast = int(np.sum(selected & (data["celltype"] == 1)))
            rows.append(
                {
                    "cohort": cohort,
                    "patient": str(patient),
                    "epithelial_cells": epithelial,
                    "fibroblast_cells": fibroblast,
                    "epithelial_qualified": epithelial >= 100,
                    "balanced_qualified": epithelial >= 100 and fibroblast >= 30,
                }
            )
pd.DataFrame(rows).to_csv(ROOT / "source_data/patient_cell_counts.csv", index=False)
print(f"Refreshed {len(mapping)+1} source tables and {len(data_mapping)} figure inputs.")
