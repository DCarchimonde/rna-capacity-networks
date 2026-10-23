"""Stream GEO gene-by-cell UMI TSVs into a five-gene cache.

This module does not infer cell types from target expression. Metadata column
names are supplied explicitly after inspecting the original metadata header.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENES = ["ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGB6"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def text_file(path):
    return (
        gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")
    )


def read_gene_tsv(path):
    """Read every gene row for the denominator; absent target rows stay missing."""
    with text_file(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cells = header[1:]
        if not cells or len(set(cells)) != len(cells) or any(not c for c in cells):
            raise ValueError("Empty or duplicate cell IDs in matrix header")
        n = len(cells)
        depth = np.zeros(n, dtype=np.int64)
        counts = np.zeros((n, 5), dtype=np.int64)
        present = np.zeros(5, dtype=bool)
        names = set()
        for row_no, row in enumerate(reader, 2):
            if len(row) != n + 1:
                raise ValueError(f"Matrix row {row_no} has the wrong number of cells")
            gene = row[0]
            if not gene or gene in names:
                raise ValueError(f"Empty/duplicate gene ID at row {row_no}: {gene}")
            names.add(gene)
            try:
                v = np.array(row[1:], dtype=np.float64)
            except ValueError as e:
                raise ValueError(f"Non-numeric counts at row {row_no}") from e
            if np.any(~np.isfinite(v)) or np.any(v < 0) or np.any(v != np.floor(v)):
                raise ValueError(f"Non-integer or negative counts at row {row_no}")
            if np.any(v > np.iinfo(np.int64).max // max(1, len(names))):
                raise ValueError("Counts exceed integer accumulator range")
            v = v.astype(np.int64)
            depth += v
            if gene in GENES:
                j = GENES.index(gene)
                counts[:, j] = v
                present[j] = True
    return np.array(cells), counts, depth, present, len(names)


def metadata_header(path):
    with text_file(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = []
        for _, row in zip(range(3), reader):
            rows.append(row)
    return {"columns": header, "first_three_rows": rows}


def prepare(metadata, manifest, column_map, counts_dir, out_dir):
    cfg = json.loads(Path(column_map).read_text())
    cols = [
        cfg[k] for k in ["cell_column", "sample_column", "main_type_column", "minor_type_column"]
    ]
    m = pd.read_csv(metadata, sep="\t", usecols=cols, dtype=str, keep_default_na=False)
    m = m.rename(columns=dict(zip(cols, ["cell_id", "sample_id", "main_type", "minor_type"])))
    if m.cell_id.duplicated().any():
        raise ValueError("Duplicate original metadata cell IDs")
    registry = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if registry.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs in fixed manifest")
    selected = registry[registry.selected == "1"].copy()
    if selected.patient_id.duplicated().any():
        raise ValueError("Manifest requires one selected specimen per patient")
    # Manifest selection is completed before counts; no expression-dependent fallback.
    patient_ids = registry.patient_id.drop_duplicates().to_numpy(dtype=str)
    records, audits, reports = [], [], []
    gene_present = np.zeros((len(patient_ids), len(GENES)), dtype=bool)
    for row in selected.itertuples(index=False):
        path = Path(counts_dir) / row.file_name
        cell_ids, x, l, present, genes = read_gene_tsv(path)
        ann = m[m.sample_id == row.sample_id].set_index("cell_id")
        ann = ann.reindex(cell_ids)
        missing_ann = ann.main_type.isna().to_numpy()
        t = np.full(len(l), -1, dtype=np.int8)
        t[ann.main_type.eq(cfg["epithelial_label"]).to_numpy()] = 0
        fibro = ann.main_type.eq(cfg["stromal_label"]) & ann.minor_type.isin(cfg["fibro_labels"])
        t[fibro.to_numpy()] = 1
        keep = (l >= cfg["minimum_UMI"]) & (t >= 0)
        report = {
            "patient": row.patient_id,
            "sample": row.sample_id,
            "matrix_cells": len(l),
            "matrix_genes": genes,
            "metadata_matched_cells": int((~missing_ann).sum()),
            "metadata_unmatched_cells": int(missing_ann.sum()),
            "annotated_cells_missing_from_matrix": int((~ann.index.isin(cell_ids)).sum()),
            "Epi_cells_UMI500": int((keep & (t == 0)).sum()),
            "Fibro_cells_UMI500": int((keep & (t == 1)).sum()),
            "all_target_rows_present": bool(present.all()),
            "missing_target_genes": "|".join(np.array(GENES)[~present]),
        }
        # Cell annotations absent from count data are reported from the original slice.
        original_ids = m.loc[m.sample_id == row.sample_id, "cell_id"]
        report["annotated_cells_missing_from_matrix"] = int((~original_ids.isin(cell_ids)).sum())
        report["primary_target_rows_present"] = bool(present[[1, 4]].all())
        report["Epi_eligible"] = bool(report["Epi_cells_UMI500"] >= 100 and present[[1, 4]].all())
        report["two_type_eligible"] = bool(
            report["Epi_eligible"] and report["Fibro_cells_UMI500"] >= 30
        )
        reports.append(report)
        audits.append(
            {
                "sample": row.sample_id,
                "file": row.file_name,
                "sha256": sha(path),
                "genes": genes,
                "target_rows_present": present.tolist(),
            }
        )
        p = np.flatnonzero(patient_ids == row.patient_id)[0]
        gene_present[p] = present
        records.append(
            (
                x[keep],
                l[keep],
                np.full(keep.sum(), p, dtype=np.int16),
                t[keep],
                cell_ids[keep],
                np.full(keep.sum(), row.sample_id),
            )
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(out / "qualification.csv", index=False)
    if not records:
        raise ValueError("No annotated cells with sufficient deposited counts")
    x, l, p, t, cells, samples = [np.concatenate([r[i] for r in records]) for i in range(6)]
    if len(set(cells)) != len(cells):
        raise ValueError("Duplicate global cell IDs across selected specimens")
    prov = {
        "source": "GSE166555",
        "kind": "independent_public_source",
        "count_layer": "GEO submitter-described raw expression counts",
        "denominator": "sum of all deposited gene rows per cell",
        "metadata_sha256": sha(metadata),
        "manifest_sha256": sha(manifest),
        "column_map_sha256": sha(column_map),
        "matrix_files": audits,
        "missing_gene_handling": "Zero storage placeholders carry a patient-by-gene presence mask; missing-edge scores are NA",
    }
    np.savez_compressed(
        out / "qualified_counts.npz",
        counts=x,
        depth=l,
        patient_index=p,
        celltype=t,
        patients=patient_ids,
        cell_id=cells,
        sample_id=samples,
        genes=np.array(GENES),
        gene_present=gene_present,
        provenance_json=np.array(json.dumps(prov)),
    )
    (out / "input_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return reports


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inspect-metadata")
    q.add_argument("metadata")
    q = sub.add_parser("prepare")
    for name in ["metadata", "manifest", "column-map", "counts-dir", "out-dir"]:
        q.add_argument("--" + name, required=True)
    a = ap.parse_args()
    if a.command == "inspect-metadata":
        print(json.dumps(metadata_header(a.metadata), ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                prepare(a.metadata, a.manifest, a.column_map, a.counts_dir, a.out_dir),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
