"""Fixed H replication pipeline. Existing development data can be replayed."""

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from estimate_h import EDGES, EDGE_NAMES, METHODS, cohort_scores, pair_metrics
from prepare_counts import GENES, sha

SCOPES = ["Epi", "Fibro", "balanced"]


def clean_json(x):
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (np.floating, float)) and not np.isfinite(x):
        return None
    return x


def analyze(cache, spec_path, out_dir, input_kind):
    clock = time.monotonic()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    spec = json.loads(Path(spec_path).read_text())
    z = np.load(cache, allow_pickle=False)
    x, l, p, t = [z[k] for k in ["counts", "depth", "patient_index", "celltype"]]
    patients = z["patients"]
    n = len(patients)
    if input_kind == "independent":
        if "provenance_json" not in z or "genes" not in z:
            raise ValueError("Independent input needs audited provenance and gene order")
        prov = json.loads(str(z["provenance_json"]))
        if prov["source"] != "GSE166555" or prov["kind"] != "independent_public_source":
            raise ValueError("Independent mode requires this fixed public source")
        if z["genes"].tolist() != GENES:
            raise ValueError("Target gene order differs from the fixed specification")
    elif sha(cache) != spec["development_cache_sha256"]:
        raise ValueError("Development replay must use the known exposed cache")
    if x.shape != (len(l), 5) or len(p) != len(l) or len(t) != len(l):
        raise ValueError("Count cache axes disagree")
    if np.any(x < 0) or np.any(x != np.floor(x)) or np.any(l != np.floor(l)):
        raise ValueError("Integer counts and all-gene depths required")
    if np.any(l < spec["minimum_UMI"]) or np.any(x.sum(axis=1) > l):
        raise ValueError("Depth qualification failed")
    if np.any(p < 0) or np.any(p >= n) or not np.isin(t, [0, 1]).all():
        raise ValueError("Invalid patient/type cache indices")
    number = np.array([[(p == i)[t == typ].sum() for typ in range(2)] for i in range(n)])
    present = z["gene_present"] if "gene_present" in z else np.ones((n, 5), dtype=bool)
    if present.shape != (n, 5):
        raise ValueError("Gene-presence mask axes disagree")
    edge_present = np.column_stack([present[:, a] & present[:, b] for a, b in EDGES])

    def score_counts(cx, cl, cp, ct):
        value = cohort_scores(cx, cl, cp, ct, n)
        value[~np.broadcast_to(edge_present[:, None, None, :], value.shape)] = np.nan
        return value

    epi = (number[:, 0] >= spec["minimum_Epi"]) & present[:, [1, 4]].all(axis=1)
    dual = epi & (number[:, 1] >= spec["minimum_Fibro"])
    eligible = [np.flatnonzero(epi), np.flatnonzero(dual), np.flatnonzero(dual)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "patient": patients,
            "Epi_cells": number[:, 0],
            "Fibro_cells": number[:, 1],
            "Epi_eligible": epi,
            "two_type_eligible": dual,
        }
    ).to_csv(out / "patient_qualification.csv", index=False)
    full = score_counts(x, l, p, t)
    records = []
    for i, patient in enumerate(patients):
        for typ, scope in enumerate(SCOPES):
            for e, edge in enumerate(EDGE_NAMES):
                h, naive, u = full[i, typ, :, e]
                if (typ == 0 and not epi[i]) or (typ > 0 and not dual[i]):
                    h = naive = u = np.nan
                records.append(
                    {
                        "patient": patient,
                        "scope": scope,
                        "edge": edge,
                        "H": h,
                        "samecell_naive": naive,
                        "crosscell_U": u,
                        "H_minus_U": h - u,
                        "edge_present": bool(edge_present[i, e]),
                    }
                )
    pd.DataFrame(records).to_csv(out / "patient_H.csv", index=False)
    splits = []
    for split in range(spec["cell_splits"]):
        rng = np.random.default_rng(spec["cell_split_seed_base"] + split)
        side = np.empty(len(l), dtype=np.int8)
        for i in range(n):
            for typ in range(2):
                ids = np.flatnonzero((p == i) & (t == typ))
                order = rng.permutation(len(ids))
                side[ids[order[: len(ids) // 2]]] = 0
                side[ids[order[len(ids) // 2 :]]] = 1
        for s in range(2):
            keep = side == s
            splits.append(score_counts(x[keep], l[keep], p[keep], t[keep]))
    halves = np.array(splits)
    split_rows, split_summary, coupling_summary = [], [], []
    for typ, scope in enumerate(SCOPES):
        for e, edge in enumerate(EDGE_NAMES):
            ix = eligible[typ][edge_present[eligible[typ], e]]
            m = len(ix)
            status = "available" if m else "missing_gene_or_no_eligible_patients"
            hfull, ufull = full[ix, typ, 0, e], full[ix, typ, 2, e]
            if m < 2:
                for name in METHODS:
                    metric = pair_metrics(np.zeros(m), np.zeros(m))
                    for role in range(len(halves)):
                        split_rows.append(
                            {
                                "scope": scope,
                                "method": name,
                                "edge": edge,
                                "role": role,
                                "status": status,
                                **metric,
                            }
                        )
                    row = {
                        "scope": scope,
                        "method": name,
                        "edge": edge,
                        "patients": m,
                        "bootstrap_valid": 0,
                        "status": status,
                    }
                    for key in [
                        "D",
                        "all_pair_retention",
                        "reference_resolution",
                        "strict_accuracy",
                        "all_pair_retention_patient_95_low",
                        "all_pair_retention_patient_95_high",
                        "reference_resolution_patient_95_low",
                        "reference_resolution_patient_95_high",
                    ]:
                        row[key] = np.nan
                    split_summary.append(row)
                coupling_summary.append(
                    {
                        "scope": scope,
                        "edge": edge,
                        "patients": m,
                        "status": status,
                        "mean_H": float(hfull.mean()) if m else np.nan,
                        "mean_U": float(ufull.mean()) if m else np.nan,
                        "H_to_U_ratio_of_means": (
                            float(hfull.mean() / ufull.mean()) if m and ufull.mean() > 0 else np.nan
                        ),
                        "mean_H_minus_U": float((hfull - ufull).mean()) if m else np.nan,
                        "patient_95_low": np.nan,
                        "patient_95_high": np.nan,
                        "positive_patients": int((hfull - ufull > 0).sum()) if m else np.nan,
                        "zero_H_patients": int((hfull == 0).sum()) if m else np.nan,
                    }
                )
                continue
            bt = np.random.default_rng(spec["patient_bootstrap_seed"]).integers(
                m, size=(spec["patient_bootstrap"], m)
            )
            bi, bj = np.triu_indices(m, 1)
            distinct = bt[:, bi] != bt[:, bj]
            den = distinct.sum(axis=1)
            good_draw = den > 0
            for method, name in enumerate(METHODS):
                h = halves[:, ix, typ, method, e]
                metrics = []
                boot_resolution = np.zeros(len(bt))
                boot_retention = np.zeros(len(bt))
                for role in range(len(h)):
                    ref, query = h[role], h[role ^ 1]
                    metric = pair_metrics(ref, query)
                    metrics.append(metric)
                    split_rows.append(
                        {
                            "scope": scope,
                            "method": name,
                            "edge": edge,
                            "role": role,
                            "status": status,
                            **metric,
                        }
                    )
                    a, b = ref[bt], query[bt]
                    d, q = a[:, bi] - a[:, bj], b[:, bi] - b[:, bj]
                    resolved = distinct & np.isfinite(d) & (d != 0)
                    kept = resolved & np.isfinite(q) & (d * q > 0)
                    np.add(
                        boot_resolution,
                        np.divide(resolved.sum(1), den, out=np.zeros(len(bt)), where=good_draw),
                        out=boot_resolution,
                    )
                    np.add(
                        boot_retention,
                        np.divide(kept.sum(1), den, out=np.zeros(len(bt)), where=good_draw),
                        out=boot_retention,
                    )
                boot_resolution /= len(h)
                boot_retention /= len(h)
                summary = {
                    "scope": scope,
                    "method": name,
                    "edge": edge,
                    "patients": m,
                    "bootstrap_valid": int(good_draw.sum()),
                    "status": status,
                }
                for key in ["D", "all_pair_retention", "reference_resolution", "strict_accuracy"]:
                    vals = np.array([r[key] for r in metrics])
                    summary[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                for key, values in [
                    ("all_pair_retention", boot_retention),
                    ("reference_resolution", boot_resolution),
                ]:
                    summary[key + "_patient_95_low"], summary[key + "_patient_95_high"] = (
                        np.quantile(values[good_draw], [0.025, 0.975])
                    )
                split_summary.append(summary)
            diff = hfull - ufull
            lo, hi = np.quantile(diff[bt].mean(axis=1), [0.025, 0.975])
            coupling_summary.append(
                {
                    "scope": scope,
                    "edge": edge,
                    "patients": m,
                    "status": status,
                    "mean_H": float(hfull.mean()),
                    "mean_U": float(ufull.mean()),
                    "H_to_U_ratio_of_means": (
                        float(hfull.mean() / ufull.mean()) if ufull.mean() > 0 else np.nan
                    ),
                    "mean_H_minus_U": float(diff.mean()),
                    "patient_95_low": lo,
                    "patient_95_high": hi,
                    "positive_patients": int((diff > 0).sum()),
                    "zero_H_patients": int((hfull == 0).sum()),
                }
            )
    pd.DataFrame(split_rows).to_csv(out / "split_roles.csv", index=False)
    pd.DataFrame(split_summary).to_csv(out / "split_summary.csv", index=False)
    pd.DataFrame(coupling_summary).to_csv(out / "coupling_summary.csv", index=False)
    bins = np.column_stack([x, l - x.sum(axis=1)])
    thin_rows, thin_scores = [], []
    low_depth_cells = 0
    for draw in range(spec["UMI_draws"]):
        rng = np.random.default_rng(spec["UMI_seed_base"] + draw)
        tx = rng.binomial(bins, spec["q"])
        tl = tx.sum(axis=1)
        keep = tl >= 2
        low_depth_cells += int((~keep).sum())
        score = score_counts(tx[keep, :5], tl[keep], p[keep], t[keep])
        thin_scores.append(score)
        for typ, scope in enumerate(SCOPES):
            ix = eligible[typ]
            for method, name in enumerate(METHODS):
                for e, edge in enumerate(EDGE_NAMES):
                    thin_rows.append(
                        {
                            "draw": draw,
                            "scope": scope,
                            "method": name,
                            "edge": edge,
                            **pair_metrics(
                                full[ix[edge_present[ix, e]], typ, method, e],
                                score[ix[edge_present[ix, e]], typ, method, e],
                            ),
                        }
                    )
    pd.DataFrame(thin_rows).to_csv(out / "UMI_roles.csv", index=False)
    np.savez_compressed(
        out / "scores.npz",
        full=full,
        halves=halves,
        thin=np.array(thin_scores),
        patients=patients,
        edges=np.array(EDGE_NAMES),
        methods=np.array(METHODS),
        scopes=np.array(SCOPES),
        gene_present=present,
        edge_present=edge_present,
    )
    result = {
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": time.monotonic() - clock,
        "input_kind": input_kind,
        "patients_in_manifest": n,
        "Epi_eligible_patients": int(epi.sum()),
        "two_type_eligible_patients": int(dual.sum()),
        "new_independent_analysis": input_kind == "independent",
        "low_depth_cells_in_UMI_draws": low_depth_cells,
        "cache_sha256": sha(cache),
        "spec_sha256": sha(spec_path),
        "script_sha256": sha(__file__),
        "gene_presence_mask": present.tolist(),
        "missing_edge_handling": "Per-edge exclusion; missing gene never reported as an observed zero",
        "intervals": "descriptive patient percentile bootstrap; technical roles do not increase biological N",
    }
    (out / "run.json").write_text(json.dumps(clean_json(result), indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ["cache", "spec", "out-dir"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--input-kind", choices=["development_replay", "independent"], required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.cache, a.spec, a.out_dir, a.input_kind), indent=2))


if __name__ == "__main__":
    main()
