"""Audit uploaded metadata and selected matrix headers before reading gene rows."""

import argparse
import csv
import datetime
import gzip
import json
from pathlib import Path

import pandas as pd
from prepare_counts import sha


def audit(metadata, archive, counts_dir, root):
    root = Path(root)
    cfg_path = root / "config/metadata_columns.json"
    cfg = json.loads(cfg_path.read_text())
    registry = pd.read_csv(root / "config/selected_tumor_manifest.csv", dtype=str)
    selected = registry.loc[registry.selected == "1"]
    with gzip.open(metadata, "rt", newline="") as f:
        header = next(csv.reader(f, delimiter="\t"))
    columns = [
        "cell",
        "cell_id",
        "sample_id",
        "case_id",
        "source_id",
        "sample_origin",
        "main_cell_type",
        "minor_cell_type",
        "cell_type_str_custom",
        "nCount_spliced",
        "nCount_unspliced",
        "nCount_ambiguous",
        "nCount_RNA",
        "nFeature_spliced",
        "nFeature_unspliced",
        "nFeature_ambiguous",
        "nFeature_RNA",
    ]
    assert set(columns).issubset(header)
    m = pd.read_csv(metadata, sep="\t", usecols=columns, dtype=str, keep_default_na=False)
    assert not m.cell_id.duplicated().any()
    assert m.cell.eq(m.cell_id).all()
    assert set(m.main_cell_type) == {"Epithelial", "Immune", "Stromal"}
    str_m = m.loc[m.main_cell_type == "Stromal"]
    assert str_m.minor_cell_type.eq(str_m.cell_type_str_custom).all()
    assert set(cfg["fibro_labels"]).issubset(set(str_m.minor_cell_type))
    records = []
    for r in selected.itertuples(index=False):
        a = m.loc[m.sample_id == r.sample_id]
        assert len(a) and set(a.sample_origin) == {"Tumor"}
        # Source authors document p001 -> p007 in their figure code.
        canonical_source = a.source_id.replace({"p001": "p007"})
        assert set(canonical_source) == {r.patient_id}
        with gzip.open(Path(counts_dir) / r.file_name, "rt", newline="") as f:
            cells = next(csv.reader(f, delimiter="\t"))[1:]
        assert len(cells) == len(set(cells)) and all(cells)
        matrix_set = set(cells)
        assert all(c.startswith(r.sample_id + ":") for c in cells)
        records.append(
            {
                "patient": r.patient_id,
                "sample": r.sample_id,
                "metadata_cells": len(a),
                "matrix_header_cells": len(cells),
                "metadata_missing_from_matrix": int((~a.cell_id.isin(matrix_set)).sum()),
                "matrix_missing_annotation": len(matrix_set - set(a.cell_id)),
                "source_ids": "|".join(sorted(a.source_id.unique())),
                "case_ids": "|".join(sorted(a.case_id.unique())),
                "Epi_before_UMI": int(a.main_cell_type.eq("Epithelial").sum()),
                "Fibro_before_UMI": int(
                    (
                        a.main_cell_type.eq("Stromal") & a.minor_cell_type.isin(cfg["fibro_labels"])
                    ).sum()
                ),
            }
        )
    # A compact projection retains annotation/count metadata, not gene expression.
    m.to_csv(
        root / "inputs/analysis_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(records).to_csv(root / "results/header_qualification.csv", index=False)
    cfg.update(
        {
            "status": "VERIFIED_GEO_HEADER_AND_DICTIONARY_BEFORE_GENE_COUNT_OPEN",
            "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "original_metadata_sha256": sha(metadata),
            "projected_metadata_sha256": sha(root / "inputs/analysis_metadata.tsv.gz"),
            "canonical_patient_mapping": "manifest sample IDs; author-documented p001 alias is p007",
        }
    )
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    audit = {
        "fixed_utc": cfg["verified_utc"],
        "status": "INPUT_MAPPING_FIXED_BEFORE_GENE_ROWS",
        "expression_gene_rows_opened": False,
        "metadata_original_rows": len(m),
        "metadata_original_columns": header,
        "main_cell_type_counts": m.main_cell_type.value_counts().to_dict(),
        "stromal_minor_type_counts": str_m.minor_cell_type.value_counts().to_dict(),
        "sample_origin_counts": m.sample_origin.value_counts().to_dict(),
        "source_count_metadata_available": [c for c in columns if c.startswith("nCount")],
        "selected_specimens": records,
        "source_files": [
            {
                "filename": Path(metadata).name,
                "bytes": Path(metadata).stat().st_size,
                "sha256": sha(metadata),
            },
            {
                "filename": Path(archive).name,
                "bytes": Path(archive).stat().st_size,
                "sha256": sha(archive),
            },
        ],
        "fixed_execution_spec_sha256": sha(root / "config/execution_spec.json"),
        "column_map_sha256": sha(cfg_path),
        "selection": "One fixed tumor per patient, p009t1 selected; p009t2 not opened.",
        "denominator_plan": "Sum all rows of deposited integer matrix; compare to each recorded layer.",
    }
    (root / "results/input_gate.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {"rows": len(m), "selected": records, "mapping_fixed_utc": cfg["verified_utc"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["metadata", "archive", "counts-dir", "root"]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    audit(a.metadata, a.archive, a.counts_dir, a.root)
