"""Exploratory resolution spectra on the two previously exposed cohorts."""

from pathlib import Path
from datetime import datetime, timezone
import time, json
import numpy as np
import pandas as pd
from resolution import conditional_flows, flows, r_from_flows, h_flows

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/execution_spec.json").read_text())
SCOPES = ["Epi", "Fibro", "balanced"]
EDGE_NAMES = ["ITGA5_ITGB1", "ITGAV_ITGB1", "ITGAV_ITGB3", "ITGAV_ITGB6"]


def aggregate(z, p, t, n):
    a = np.full((n, 3, *z.shape[1:]), np.nan)
    for patient in range(n):
        for typ in range(2):
            select = (p == patient) & (t == typ)
            if select.any():
                a[patient, typ] = z[select].mean(0)
        a[patient, 2] = (a[patient, 0] + a[patient, 1]) / 2
    return a


def pair_counts(a, b):
    i, j = np.triu_indices(len(a), 1)
    da, db = a[i] - a[j], b[i] - b[j]
    finite = np.isfinite(da) & np.isfinite(db)
    resolved = finite & (da != 0) & (db != 0)
    same = resolved & (da * db > 0)
    opposite = resolved & (da * db < 0)
    return {
        "all_pairs": len(i),
        "same_pairs": int(same.sum()),
        "opposite_pairs": int(opposite.sum()),
        "unresolved_pairs": int((~resolved).sum()),
        "all_pair_retention": float(same.mean()) if len(i) else None,
        "reference_resolution": float((np.isfinite(da) & (da != 0)).mean()) if len(i) else None,
    }


def run_one(name, filename, common):
    start = time.monotonic()
    z = np.load(ROOT / "inputs" / filename, allow_pickle=False)
    x, l, p, t = [z[j] for j in ["counts", "depth", "patient_index", "celltype"]]
    patients = z["patients"]
    n = len(patients)
    ks = SPEC["resolutions_real"]
    graph = "common3" if common else "full4"
    if not common and "gene_present" in z and not z["gene_present"].all():
        raise ValueError("Full graph requires all five genes")
    cf, errors = conditional_flows(x, l, ks, common)
    h = h_flows(x, l, common)
    k2err = float(np.max(np.abs(cf[:, 0] - h)))
    assert k2err < 1e-12
    plug = flows(x / l[:, None], common)
    mininc = float(np.diff(cf, axis=1).min())
    maxjensen = float(np.max(cf - plug[:, None, :]))
    assert mininc > -1e-10 and maxjensen < 1e-10
    allflow = aggregate(cf, p, t, n)
    allr = r_from_flows(allflow)
    plugflow = aggregate(plug, p, t, n)
    plugr = r_from_flows(plugflow)
    counts = np.array([[((p == i) & (t == typ)).sum() for typ in range(2)] for i in range(n)])
    epi = counts[:, 0] >= SPEC["minimum_Epi"]
    dual = epi & (counts[:, 1] >= SPEC["minimum_Fibro"])
    eligible = [np.flatnonzero(epi), np.flatnonzero(dual), np.flatnonzero(dual)]
    splitflows = []
    for split in range(SPEC["cell_splits"]):
        rng = np.random.default_rng(SPEC["cell_split_seed_base"] + split)
        side = np.empty(len(l), np.int8)
        for i in range(n):
            for typ in range(2):
                ids = np.flatnonzero((p == i) & (t == typ))
                order = rng.permutation(len(ids))
                side[ids[order[: len(ids) // 2]]] = 0
                side[ids[order[len(ids) // 2 :]]] = 1
        for s in range(2):
            keep = side == s
            splitflows.append(aggregate(cf[keep], p[keep], t[keep], n))
    splitflows = np.array(splitflows)
    splitr = r_from_flows(splitflows)
    records = []
    summary = []
    role_rows = []
    transition = []
    en = [EDGE_NAMES[j] for j in ([0, 1, 3] if common else range(4))]
    for typ, scope in enumerate(SCOPES):
        ix = eligible[typ]
        for q, k in enumerate(ks):
            rr = allr[ix, typ, q]
            for i in ix:
                row = {
                    "cohort": name,
                    "graph": graph,
                    "patient": str(patients[i]),
                    "scope": scope,
                    "k": k,
                    "R_k": float(allr[i, typ, q]),
                    "bottleneck": en[int(allflow[i, typ, q, 1:].argmin())],
                    "depth_varying_plugin_R": float(plugr[i, typ]),
                    "Epi_cells": int(counts[i, 0]),
                    "Fibro_cells": int(counts[i, 1]),
                }
                for j, v in enumerate(allflow[i, typ, q]):
                    row["A_" + ("0" if j == 0 else en[j - 1])] = v
                records.append(row)
            metrics = []
            for role in range(len(splitr)):
                m = pair_counts(splitr[role, ix, typ, q], splitr[role ^ 1, ix, typ, q])
                metrics.append(m)
                role_rows.append(
                    {"cohort": name, "graph": graph, "scope": scope, "k": k, "role": role, **m}
                )
            summary.append(
                {
                    "cohort": name,
                    "graph": graph,
                    "scope": scope,
                    "k": k,
                    "patients": len(ix),
                    "R_median": np.nanmedian(rr),
                    "R_min": np.nanmin(rr),
                    "R_max": np.nanmax(rr),
                    "positive_baseline_patients": int((allflow[ix, typ, q, 0] > 0).sum()),
                    "split_all_pair_retention": np.mean([m["all_pair_retention"] for m in metrics]),
                    "split_reference_resolution": np.mean(
                        [m["reference_resolution"] for m in metrics]
                    ),
                }
            )
            for q2 in range(q + 1, len(ks)):
                pc = pair_counts(rr, allr[ix, typ, q2])
                transition.append(
                    {
                        "cohort": name,
                        "graph": graph,
                        "scope": scope,
                        "k_from": k,
                        "k_to": ks[q2],
                        "patients": len(ix),
                        **pc,
                        "bottleneck_switch_patients": int(
                            (
                                allflow[ix, typ, q, 1:].argmin(1)
                                != allflow[ix, typ, q2, 1:].argmin(1)
                            ).sum()
                        ),
                    }
                )
        pc = pair_counts(allr[ix, typ, -1], plugr[ix, typ])
        transition.append(
            {
                "cohort": name,
                "graph": graph,
                "scope": scope,
                "k_from": ks[-1],
                "k_to": "observed_L",
                "patients": len(ix),
                **pc,
                "bottleneck_switch_patients": int(
                    (allflow[ix, typ, -1, 1:].argmin(1) != plugflow[ix, typ, 1:].argmin(1)).sum()
                ),
            }
        )
    np.savez_compressed(
        ROOT / "results" / f"{name}_{graph}_cell_flows.npz",
        conditional_flows=cf,
        depth_varying_plugin_flow=plug,
        patient_flow=allflow,
        patient_R=allr,
        split_flow=splitflows,
        split_R=splitr,
        patients=patients,
        k=np.array(ks),
    )
    return (
        records,
        summary,
        role_rows,
        transition,
        {
            "cohort": name,
            "graph": graph,
            "cells": len(l),
            "seconds": time.monotonic() - start,
            "max_probability_mass_error": float(errors.max()),
            "k2_vs_factorial_max_error": k2err,
            "minimum_flow_increment": mininc,
            "max_conditional_Jensen_violation": maxjensen,
        },
    )


def main():
    started = datetime.now(timezone.utc).isoformat()
    tables = [[], [], [], []]
    checks = []
    for name, file, common in [
        ("D", "development_counts.npz", False),
        ("D", "development_counts.npz", True),
        ("I", "independent_counts.npz", True),
    ]:
        result = run_one(name, file, common)
        for j in range(4):
            tables[j].extend(result[j])
        checks.append(result[4])
        print(json.dumps(result[4]), flush=True)
    for rows, name in zip(
        tables, ["patient_spectrum", "cohort_summary", "split_roles", "resolution_transitions"]
    ):
        pd.DataFrame(rows).to_csv(ROOT / "results" / f"{name}.csv", index=False)
    result = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-replication exploratory fixed-resolution calculation",
        "checks": checks,
        "all_checks_passed": True,
    }
    (ROOT / "results/cohort_run.json").write_text(json.dumps(result, indent=2) + "\n")
    t = pd.DataFrame(tables[3])
    print(t[(t.k_from == 2) & (t.k_to == 500)].to_string(index=False))


if __name__ == "__main__":
    main()
