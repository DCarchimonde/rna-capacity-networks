"""Run unchanged count reader while auditing depth against author count layers."""

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import prepare_counts as core


def run(root, counts_dir):
    root = Path(root)
    gate = json.loads((root / "results/input_gate.json").read_text())
    assert gate["expression_gene_rows_opened"] is False
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    m = pd.read_csv(
        root / "inputs/analysis_metadata.tsv.gz", sep="\t", dtype=str, keep_default_na=False
    ).set_index("cell_id")
    count_columns = ["nCount_spliced", "nCount_unspliced", "nCount_ambiguous", "nCount_RNA"]
    layer_audit = []
    read_original = core.read_gene_tsv

    def read_and_audit(path):
        clock = time.monotonic()
        result = read_original(path)
        cells, _, depth, present, genes = result
        a = m.reindex(cells)
        matched = a.sample_id.notna().to_numpy()
        candidates = {c: pd.to_numeric(a[c], errors="coerce").to_numpy() for c in count_columns}
        candidates["spliced_plus_unspliced"] = (
            candidates["nCount_spliced"] + candidates["nCount_unspliced"]
        )
        candidates["spliced_plus_unspliced_plus_ambiguous"] = (
            candidates["spliced_plus_unspliced"] + candidates["nCount_ambiguous"]
        )
        for name, values in candidates.items():
            valid = matched & np.isfinite(values)
            delta = depth[valid] - values[valid]
            layer_audit.append(
                {
                    "file": Path(path).name,
                    "layer_candidate": name,
                    "matrix_cells": len(cells),
                    "matrix_gene_rows": genes,
                    "available_matched_cells": int(valid.sum()),
                    "exact_matches": int((delta == 0).sum()),
                    "exact_match_fraction": float((delta == 0).mean()) if len(delta) else None,
                    "max_abs_delta": float(np.abs(delta).max()) if len(delta) else None,
                    "all_target_rows_present": bool(present.all()),
                }
            )
        print(
            json.dumps(
                {
                    "file": Path(path).name,
                    "cells": len(cells),
                    "genes": genes,
                    "seconds": round(time.monotonic() - clock, 2),
                    "targets_present": present.tolist(),
                }
            ),
            flush=True,
        )
        pd.DataFrame(layer_audit).to_csv(root / "results/count_layer_audit.csv", index=False)
        return result

    core.read_gene_tsv = read_and_audit
    core.prepare(
        root / "inputs/analysis_metadata.tsv.gz",
        root / "config/selected_tumor_manifest.csv",
        root / "config/metadata_columns.json",
        counts_dir,
        root / "inputs/prepared",
    )
    a = pd.DataFrame(layer_audit)
    exact_layers = []
    for name in a.layer_candidate.unique():
        v = a.loc[a.layer_candidate == name]
        if (
            len(v) == 12
            and v.available_matched_cells.gt(0).all()
            and v.exact_match_fraction.eq(1).all()
        ):
            exact_layers.append(name)
    # Identity is established by all cell totals and the source's preprocessing,
    # independently of H or any patient ranking outcome.
    if "nCount_spliced" in exact_layers:
        layer = (
            "Deposited nonnegative integer counts consistent with the author's spliced layer; "
            "every matched all-gene total equals nCount_spliced (and the RNA alias if listed)"
        )
    else:
        layer = (
            "Deposited nonnegative integer raw-count layer; exact-matching metadata totals: "
            + "|".join(exact_layers)
        )
    prepared = root / "inputs/prepared"
    prov = json.loads((prepared / "input_provenance.json").read_text())
    prov.update(
        {
            "count_layer": layer,
            "exact_matching_metadata_count_layers": exact_layers,
            "original_uploaded_sources": gate["source_files"],
            "count_preparation_started_utc": started,
            "count_preparation_finished_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "input_gate_sha256": core.sha(root / "results/input_gate.json"),
            "count_layer_audit_sha256": core.sha(root / "results/count_layer_audit.csv"),
            "preparation_core_sha256": core.sha(core.__file__),
            "preparation_wrapper_sha256": core.sha(__file__),
        }
    )
    cache = prepared / "qualified_counts.npz"
    with np.load(cache, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["provenance_json"] = np.array(json.dumps(prov))
    np.savez_compressed(cache, **data)
    (prepared / "input_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(
        json.dumps(
            {
                "count_layer": layer,
                "cache_sha256": core.sha(cache),
                "qualification": pd.read_csv(prepared / "qualification.csv").to_dict("records"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--counts-dir", required=True)
    a = p.parse_args()
    run(a.root, a.counts_dir)
