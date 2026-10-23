"""Replay all old H cohorts, add compatible Poisson controls, and fit both CRC sources."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import hashlib, json, time, warnings
import numpy as np
import pandas as pd
from cscore_adapter import METHODS, summarize_type, r2_from_h
from cell_locality_reference import true_patient
from numeric_metrics import pair_metrics

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/comparison_spec.json").read_text())
SCOPES = ["Epi", "Fibro", "balanced"]
EDGE_NAMES = ["ITGA5_ITGB1", "ITGAV_ITGB1", "ITGAV_ITGB3", "ITGAV_ITGB6"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_cohort(task):
    source, c, rep, case, seed = task
    rng = np.random.default_rng(seed + c * 1000 + rep)
    coordinate = rng.permutation(np.linspace(0, 1, 23))
    typ = ["Epi", "Fibro"].index(case["signal_type"])
    xx = []
    ll = []
    ss = []
    truth = []
    ids = []
    for z in coordinate:
        fam = case["family"]
        rho = 0.4 if fam in ["mean_only", "null"] else 0.1 + 0.8 * z
        delta = 0 if fam == "coupling_only" else (0.4 if fam == "null" else 0.8 * z)
        _, _, _, states, weights = true_patient(rho, delta, typ)
        patient_x = []
        patient_l = []
        patient_sid = []
        patient_truth = []
        for t, n in enumerate([200, 100]):
            sid = rng.choice(5, size=n, p=weights[t])
            p = states[t][sid]
            if source == "multinomial":
                depth = np.full(n, 1000 if case["depth_design"] == "low" else 16000, dtype=int)
                if case["depth_design"] == "state_dependent":
                    depth = np.array([8000, 16000, 2000, 2000, 2000])[sid]
                counts = rng.multinomial(depth, np.column_stack([p, 1 - p.sum(1)]))[:, :5]
            else:
                if case["depth_design"] == "constant_1000":
                    depth = np.full(n, 1000.0)
                elif case["depth_design"] == "independent_variable":
                    depth = np.exp(rng.uniform(np.log(1000.0), np.log(16000.0), n))
                else:
                    depth = np.array([8000.0, 16000.0, 2000.0, 2000.0, 2000.0])[sid]
                counts = rng.poisson(depth[:, None] * p)
            h = np.array(
                [np.dot(weights[t], states[t][:, a] * states[t][:, b]) for a, b in SPEC["edges"]]
            )
            patient_x.append(counts)
            patient_l.append(depth)
            patient_sid.append(sid)
            patient_truth.append(h)
        xx.append(patient_x)
        ll.append(patient_l)
        ss.append(patient_sid)
        truth.append([*patient_truth, (patient_truth[0] + patient_truth[1]) / 2])
    return coordinate, xx, ll, ss, np.asarray(truth)


def simulate(task):
    source, c, rep, case, seed = task
    coordinate, xx, ll, ss, truth = make_cohort(task)
    if source == "multinomial" and rep == 0:
        source_data = np.load(ROOT / f"inputs/simulation_source_case_{c}.npz")
        assert np.array_equal(coordinate, source_data["coordinate"])
        for typ, label in enumerate(["E", "F"]):
            assert np.array_equal(np.array([a[typ] for a in xx]), source_data[label + "_counts"])
            assert np.array_equal(np.array([a[typ] for a in ll]), source_data[label + "_depth"])
            assert np.array_equal(np.array([a[typ] for a in ss]), source_data[label + "_states"])
    values = np.full((23, 3, len(METHODS), 4), np.nan)
    status = []
    for i in range(23):
        for t in range(2):
            values[i, t], records = summarize_type(xx[i][t], ll[i][t], poisson=source == "poisson")
            status.extend(
                [
                    {
                        "source": source,
                        "case": c,
                        "replicate": rep,
                        **case,
                        "patient_index": i,
                        "cell_type": SCOPES[t],
                        **s,
                    }
                    for s in records
                ]
            )
        values[i, 2] = (values[i, 0] + values[i, 1]) / 2
    rtruth = r2_from_h(truth)
    rvalue = r2_from_h(values)
    rows = []
    for scope_i, scope in enumerate(SCOPES):
        for mi, method in enumerate(METHODS):
            for ei, edge in enumerate(EDGE_NAMES):
                if mi == 5 and ei != 3:
                    continue
                ref, q = truth[:, scope_i, ei], values[:, scope_i, mi, ei]
                finite = np.isfinite(q)
                errors = q - ref
                row = {
                    "source": source,
                    "case": c,
                    "replicate": rep,
                    **case,
                    "scope": scope,
                    "method": method,
                    "target": edge,
                    "target_kind": "H",
                    **pair_metrics(ref, q),
                    "mean_truth": float(ref.mean()),
                    "mean_estimate": float(np.nanmean(q)) if finite.any() else np.nan,
                    "mean_bias": float(np.nanmean(errors)) if finite.any() else np.nan,
                    "RMSE": float(np.sqrt(np.nanmean(errors**2))) if finite.any() else np.nan,
                    "MAE": float(np.nanmean(abs(errors))) if finite.any() else np.nan,
                    "point_coverage": float(finite.mean()),
                    "negative_estimates": int((q < 0).sum()),
                }
                rows.append(row)
            if mi == 5:
                continue
            ref, q = rtruth[:, scope_i], rvalue[:, scope_i, mi]
            finite = np.isfinite(q)
            err = q - ref
            rows.append(
                {
                    "source": source,
                    "case": c,
                    "replicate": rep,
                    **case,
                    "scope": scope,
                    "method": method,
                    "target": "R2_full4",
                    "target_kind": "R2",
                    **pair_metrics(ref, q),
                    "mean_truth": float(ref.mean()),
                    "mean_estimate": float(np.nanmean(q)) if finite.any() else np.nan,
                    "mean_bias": float(np.nanmean(err)) if finite.any() else np.nan,
                    "RMSE": float(np.sqrt(np.nanmean(err**2))) if finite.any() else np.nan,
                    "MAE": float(np.nanmean(abs(err))) if finite.any() else np.nan,
                    "point_coverage": float(finite.mean()),
                    "negative_estimates": 0,
                }
            )
    return rows, status, values, truth


def run_simulations():
    start = time.monotonic()
    plan = json.loads((ROOT / "inputs/original_H_experiment_plan.json").read_text())
    tasks = [
        ("multinomial", c, r, case, plan["seed_base"])
        for c, case in enumerate(plan["cases"])
        for r in range(32)
    ]
    pc = SPEC["additional_poisson_control"]
    cases = [
        {"family": f, "signal_type": t, "depth_design": d}
        for f in pc["families"]
        for t in pc["signal_types"]
        for d in pc["depth_designs"]
    ]
    tasks += [
        ("poisson", c, r, case, pc["seed_base"])
        for c, case in enumerate(cases)
        for r in range(pc["replicates_per_case"])
    ]
    rows = []
    fits = []
    values = []
    truth = []
    old = np.load(ROOT / "inputs/original_simulation_scores.npz")
    old_maxerr = 0.0
    with ProcessPoolExecutor(max_workers=4) as pool:
        for k, result in enumerate(pool.map(simulate, tasks, chunksize=4)):
            row, status, v, h = result
            rows.extend(row)
            fits.extend(status)
            values.append(v)
            truth.append(h)
            if k < 768:
                err = float(np.max(np.abs(v[:, 2, 0] - old["H_estimates"][k, :, 4])))
                old_maxerr = max(old_maxerr, err)
                assert np.allclose(v[:, 2, 0], old["H_estimates"][k, :, 4], rtol=1e-12, atol=1e-22)
                assert np.allclose(h[:, 2], old["H_truth"][k], rtol=1e-12, atol=1e-22)
            if (k + 1) % 96 == 0:
                print(
                    json.dumps(
                        {
                            "H_simulations_done": k + 1,
                            "total": len(tasks),
                            "seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(ROOT / "results/H_simulation_metrics.csv.gz", index=False)
    pd.DataFrame(fits).to_csv(ROOT / "results/H_simulation_fit_status.csv.gz", index=False)
    np.savez_compressed(
        ROOT / "results/H_simulation_scores.npz",
        values=np.array(values),
        truth=np.array(truth),
        methods=np.array(METHODS),
        scopes=np.array(SCOPES),
        edges=np.array(EDGE_NAMES),
        source=np.array([t[0] for t in tasks]),
        cases=np.array([t[1] for t in tasks]),
        replicates=np.array([t[2] for t in tasks]),
    )
    result = {
        "cohorts": len(tasks),
        "old_cohorts_replayed": 768,
        "new_poisson_cohorts": 384,
        "all_24_source_snapshots_identical": True,
        "old_H_score_max_error": old_maxerr,
        "simulated_cells": len(tasks) * 23 * 300,
        "fit_calls": len(fits),
        "seconds": time.monotonic() - start,
    }
    (ROOT / "results/H_simulation_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def run_real(name, file):
    start = time.monotonic()
    z = np.load(ROOT / "inputs" / file, allow_pickle=False)
    x, l, p, t = [z[key] for key in ["counts", "depth", "patient_index", "celltype"]]
    patients = z["patients"]
    n = len(patients)
    present = z["gene_present"] if "gene_present" in z else np.ones((n, 5), bool)
    number = np.array([[((p == i) & (t == j)).sum() for j in range(2)] for i in range(n)])
    epi = number[:, 0] >= 100
    dual = epi & (number[:, 1] >= 30)
    eligible = [np.flatnonzero(epi), np.flatnonzero(dual), np.flatnonzero(dual)]
    fits = []

    def score(cx, cl, cp, ct, role, kind):
        values = np.full((n, 3, len(METHODS), 4), np.nan)
        for i in range(n):
            for typ in range(2):
                if (typ == 0 and not epi[i]) or (typ == 1 and not dual[i]):
                    continue
                keep = (cp == i) & (ct == typ)
                values[i, typ], records = summarize_type(cx[keep], cl[keep], present[i])
                fits.extend(
                    [
                        {
                            "cohort": name,
                            "role": role,
                            "kind": kind,
                            "patient": str(patients[i]),
                            "cell_type": SCOPES[typ],
                            **s,
                        }
                        for s in records
                    ]
                )
            values[i, 2] = (values[i, 0] + values[i, 1]) / 2
        return values

    full = score(x, l, p, t, 0, "full")
    halves = []
    for split in range(SPEC["real_data"]["cell_splits"]):
        rng = np.random.default_rng(SPEC["real_data"]["cell_split_seed_base"] + split)
        side = np.empty(len(l), np.int8)
        for i in range(n):
            for typ in range(2):
                ids = np.flatnonzero((p == i) & (t == typ))
                order = rng.permutation(len(ids))
                side[ids[order[: len(ids) // 2]]] = 0
                side[ids[order[len(ids) // 2 :]]] = 1
        for s in range(2):
            keep = side == s
            halves.append(score(x[keep], l[keep], p[keep], t[keep], 2 * split + s, "cell_half"))
    halves = np.array(halves)
    thin = []
    bins = np.column_stack([x, l - x.sum(1)])
    for draw in range(SPEC["real_data"]["UMI_draws"]):
        rng = np.random.default_rng(SPEC["real_data"]["UMI_seed_base"] + draw)
        tx = rng.binomial(bins, 0.5)
        tl = tx.sum(1)
        keep = tl >= 2
        thin.append(score(tx[keep, :5], tl[keep], p[keep], t[keep], draw, "UMI_thin"))
        if (draw + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "cohort": name,
                        "UMI_draws_done": draw + 1,
                        "seconds": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )
    thin = np.array(thin)
    np.savez_compressed(
        ROOT / f"results/{name}_H_scores.npz",
        full=full,
        halves=halves,
        thin=thin,
        patients=patients,
        methods=np.array(METHODS),
        scopes=np.array(SCOPES),
        edges=np.array(EDGE_NAMES),
        gene_present=present,
        patient_cell_counts=number,
        Epi_eligible=epi,
        dual_eligible=dual,
    )
    pd.DataFrame(fits).to_csv(ROOT / f"results/{name}_fit_status.csv.gz", index=False)
    # Same-target real comparisons do not label repeatability as latent accuracy.
    records = []
    metrics = []
    agreement = []
    for typ, scope in enumerate(SCOPES):
        ix = eligible[typ]
        for mi, method in enumerate(METHODS):
            for e, edge in enumerate(EDGE_NAMES):
                if (name == "I" and e == 2) or (mi == 5 and e != 3):
                    continue
                for i in ix:
                    records.append(
                        {
                            "cohort": name,
                            "patient": str(patients[i]),
                            "scope": scope,
                            "method": method,
                            "target": edge,
                            "H": full[i, typ, mi, e],
                        }
                    )
                for role in range(len(halves)):
                    metrics.append(
                        {
                            "cohort": name,
                            "scope": scope,
                            "method": method,
                            "target": edge,
                            "kind": "cell_split",
                            "role": role,
                            **pair_metrics(
                                halves[role, ix, typ, mi, e], halves[role ^ 1, ix, typ, mi, e]
                            ),
                        }
                    )
                for role in range(len(thin)):
                    metrics.append(
                        {
                            "cohort": name,
                            "scope": scope,
                            "method": method,
                            "target": edge,
                            "kind": "UMI_thin",
                            "role": role,
                            **pair_metrics(full[ix, typ, mi, e], thin[role, ix, typ, mi, e]),
                        }
                    )
                agreement.append(
                    {
                        "cohort": name,
                        "scope": scope,
                        "method": method,
                        "target": edge,
                        **pair_metrics(full[ix, typ, 0, e], full[ix, typ, mi, e]),
                        "mean_equal_cell_H": float(np.mean(full[ix, typ, 0, e])),
                        "mean_method_H": float(np.nanmean(full[ix, typ, mi, e])),
                    }
                )
    pd.DataFrame(records).to_csv(ROOT / f"results/{name}_patient_H.csv", index=False)
    pd.DataFrame(metrics).to_csv(ROOT / f"results/{name}_H_roles.csv.gz", index=False)
    pd.DataFrame(agreement).to_csv(ROOT / f"results/{name}_H_agreement.csv", index=False)
    result = {
        "cohort": name,
        "Epi_patients": int(epi.sum()),
        "dual_patients": int(dual.sum()),
        "cells": len(l),
        "fits": len(fits),
        "seconds": time.monotonic() - start,
        "input_sha256": sha(ROOT / "inputs" / file),
    }
    (ROOT / f"results/{name}_H_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    started = datetime.now(timezone.utc).isoformat()
    run_simulations()
    for name, file in [("D", "development_counts.npz"), ("I", "independent_counts.npz")]:
        run_real(name, file)
    (ROOT / "results/H_execution.json").write_text(
        json.dumps(
            {
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "spec_sha256": sha(ROOT / "config/comparison_spec.json"),
                "official_source_sha256": sha(ROOT / "vendor/CSCORE_IRLS.py"),
                "benchmark_sha256": sha(__file__),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
