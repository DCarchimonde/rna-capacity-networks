"""Exact fixed-resolution plug-in against exact rarefaction, same R8 target."""

from pathlib import Path
import sys

# Reuse the canonical fixed-resolution implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "molecular_resolution/scripts"))
from datetime import datetime, timezone
import json, time
import numpy as np
import pandas as pd
from plugin import multinomial_flows
from resolution import conditional_flows, r_from_flows
from run_simulations import SPEC, to_arrays, draw_counts, known_flows, agg
from verify_theory import mean_matched_pop, second_moment_pop
from analyze_cohorts import aggregate, pair_counts

ROOT = Path(__file__).resolve().parents[1]


def simulation():
    start = time.monotonic()
    n = SPEC["patients"]
    ne, nf = SPEC["Epi"], SPEC["Fibro"]
    m = ne + nf
    k = SPEC["k"]
    eta = np.linspace(0.1, 0.9, n)
    patient = np.repeat(np.arange(n), m)
    typ = np.tile(np.r_[np.zeros(ne, int), np.ones(nf, int)], n)
    half = np.tile(np.r_[np.arange(ne) % 2, np.arange(nf) % 2], n)
    old = pd.read_csv(ROOT / "../molecular_resolution/results/simulation_patient_scores.csv")
    metrics = []
    scores = []
    max_replay = 0.0
    maxmass = 0.0
    for fi, family in enumerate(SPEC["families"]):
        target = known_flows(family, k, eta)
        trueR = r_from_flows(target)
        for di, L in enumerate(SPEC["depths"]):
            for rep in range(SPEC["cohort_replicates"]):
                seed = SPEC["seed_base"] + fi * 100000 + di * 1000 + rep
                rng = np.random.default_rng(seed)
                if family == "identical_biology_depth_gradient":
                    latent = np.tile([0.06, 0.08, 0.1, 0.03, 0.05], (len(patient), 1))
                    dpatient = np.rint(np.geomspace(L, 4 * L, n)).astype(np.int64)
                else:
                    pa, pb = (
                        (mean_matched_pop(0), mean_matched_pop(1))
                        if family == "mean_matched_joint"
                        else (second_moment_pop(0), second_moment_pop(1))
                    )
                    wa, aa = to_arrays(pa)
                    wb, bb = to_arrays(pb)
                    choose_b = rng.random(len(patient)) < eta[patient]
                    latent = np.empty((len(patient), 5))
                    for flag, w, states in [(False, wa, aa), (True, wb, bb)]:
                        keep = choose_b == flag
                        latent[keep] = states[rng.choice(len(w), keep.sum(), p=w)]
                    dpatient = np.full(n, L, dtype=np.int64)
                depth = dpatient[patient]
                x = draw_counts(latent, depth, rng)
                exact, err = conditional_flows(x, depth, [k])
                exact = exact[:, 0]
                expected = r_from_flows(agg(exact, patient, typ, n))
                prior = (
                    old[
                        (old.family == family)
                        & (old.base_depth == L)
                        & (old.rep == rep)
                        & (old.method == "exact_conditional")
                    ]
                    .sort_values("patient")
                    .R.to_numpy()
                )
                replay = float(np.max(abs(expected - prior)))
                max_replay = max(max_replay, replay)
                assert replay < 1e-11
                plugin, mass = multinomial_flows(x / depth[:, None], k)
                maxmass = max(maxmass, mass.max())
                af = agg(plugin, patient, typ, n)
                rr = r_from_flows(af)
                r0 = r_from_flows(agg(plugin[half == 0], patient[half == 0], typ[half == 0], n))
                r1 = r_from_flows(agg(plugin[half == 1], patient[half == 1], typ[half == 1], n))
                retained = pair_counts(r0, r1)
                isnull = family == "identical_biology_depth_gradient"
                metrics.append(
                    {
                        "family": family,
                        "base_depth": L,
                        "rep": rep,
                        "method": "exact_multinomial_plugin",
                        "flow_MSE_R8_target": np.mean((af - target) ** 2),
                        "R8_MSE": np.nanmean((rr - trueR) ** 2),
                        "R8_pair_accuracy": (
                            pair_counts(trueR, rr)["all_pair_retention"] if not isnull else np.nan
                        ),
                        "split_all_pair_retention": retained["all_pair_retention"],
                        "split_reference_resolution": retained["reference_resolution"],
                        "finite_patients": int(np.isfinite(rr).sum()),
                        "R_depth_correlation": (
                            np.corrcoef(dpatient, rr)[0, 1] if isnull else np.nan
                        ),
                    }
                )
                scores.extend(
                    [
                        {
                            "family": family,
                            "base_depth": L,
                            "rep": rep,
                            "method": "exact_multinomial_plugin",
                            "patient": i,
                            "depth": dpatient[i],
                            "R": rr[i],
                            "true_R8": trueR[i],
                            "A0": af[i, 0],
                            "A0_true": target[i, 0],
                        }
                        for i in range(n)
                    ]
                )
            print(
                json.dumps(
                    {
                        "R8_done": family,
                        "base_depth": L,
                        "seconds": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )
    merged = pd.concat(
        [
            pd.read_csv(ROOT / "../molecular_resolution/results/simulation_replicates.csv"),
            pd.DataFrame(metrics),
        ],
        ignore_index=True,
    )
    merged.to_csv(ROOT / "results/R8_simulation_replicates.csv", index=False)
    pd.DataFrame(scores).to_csv(ROOT / "results/R8_plugin_patient_scores.csv", index=False)
    merged.groupby(["family", "base_depth", "method"], sort=False).mean(
        numeric_only=True
    ).reset_index().to_csv(ROOT / "results/R8_simulation_summary.csv", index=False)
    gains = []
    for (family, L), g in merged.groupby(["family", "base_depth"]):
        a = g[g.method == "exact_conditional"].set_index("rep")
        b = g[g.method == "exact_multinomial_plugin"].set_index("rep")
        for endpoint in ["R8_pair_accuracy", "R8_MSE", "flow_MSE_R8_target"]:
            diff = (a[endpoint] - b[endpoint]).dropna().to_numpy()
            if len(diff):
                rng = np.random.default_rng(2610109292)
                ix = rng.integers(len(diff), size=(5000, len(diff)))
                lo, hi = np.quantile(diff[ix].mean(1), [0.025, 0.975])
                gains.append(
                    {
                        "family": family,
                        "base_depth": L,
                        "endpoint": endpoint,
                        "exact_minus_plugin": diff.mean(),
                        "MC_95_low": lo,
                        "MC_95_high": hi,
                    }
                )
    pd.DataFrame(gains).to_csv(ROOT / "results/R8_paired_comparison.csv", index=False)
    result = {
        "replayed_cohorts": 192,
        "replay_R_max_error": max_replay,
        "plugin_probability_mass_max_error": float(maxmass),
        "seconds": time.monotonic() - start,
    }
    (ROOT / "results/R8_simulation_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def real():
    rows = []
    summary = []
    maxmass = 0
    for name, file, common in [
        ("D", "development_counts.npz", True),
        ("I", "independent_counts.npz", True),
        ("D", "development_counts.npz", False),
    ]:
        z = np.load(ROOT / "inputs" / file)
        x, l, p, t = [z[key] for key in ["counts", "depth", "patient_index", "celltype"]]
        patients = z["patients"]
        n = len(patients)
        graph = "common3" if common else "full4"
        plugin, err = multinomial_flows(x / l[:, None], 8, common)
        maxmass = max(maxmass, err.max())
        af = aggregate(plugin, p, t, n)
        r = r_from_flows(af)
        half = []
        old = np.load(ROOT / f"../molecular_resolution/results/{name}_{graph}_cell_flows.npz")
        prior = old["patient_R"][:, :, 1]
        priorhalf = old["split_R"][:, :, :, 1]
        number = np.array([[((p == i) & (t == j)).sum() for j in range(2)] for i in range(n)])
        eligible = [np.flatnonzero(number[:, 0] >= 100)] + [
            np.flatnonzero((number[:, 0] >= 100) & (number[:, 1] >= 30))
        ] * 2
        for split in range(20):
            rng = np.random.default_rng(4402000 + split)
            side = np.empty(len(l), np.int8)
            for i in range(n):
                for typ in range(2):
                    ids = np.flatnonzero((p == i) & (t == typ))
                    order = rng.permutation(len(ids))
                    side[ids[order[: len(ids) // 2]]] = 0
                    side[ids[order[len(ids) // 2 :]]] = 1
            for s in range(2):
                keep = side == s
                half.append(r_from_flows(aggregate(plugin[keep], p[keep], t[keep], n)))
        half = np.array(half)
        for typ, scope in enumerate(["Epi", "Fibro", "balanced"]):
            ix = eligible[typ]
            agreement = pair_counts(prior[ix, typ], r[ix, typ])
            split = [
                pair_counts(half[role, ix, typ], half[role ^ 1, ix, typ]) for role in range(40)
            ]
            oldsplit = [
                pair_counts(priorhalf[role, ix, typ], priorhalf[role ^ 1, ix, typ])
                for role in range(40)
            ]
            summary.append(
                {
                    "cohort": name,
                    "graph": graph,
                    "scope": scope,
                    "patients": len(ix),
                    **agreement,
                    "exact_split_retention": np.mean([a["all_pair_retention"] for a in oldsplit]),
                    "plugin_split_retention": np.mean([a["all_pair_retention"] for a in split]),
                    "R_mean_exact": np.mean(prior[ix, typ]),
                    "R_mean_plugin": np.mean(r[ix, typ]),
                    "R_max_abs_difference": np.max(abs(prior[ix, typ] - r[ix, typ])),
                }
            )
            for i in ix:
                rows.append(
                    {
                        "cohort": name,
                        "graph": graph,
                        "scope": scope,
                        "patient": str(patients[i]),
                        "R8_exact": prior[i, typ],
                        "R8_plugin": r[i, typ],
                    }
                )
    pd.DataFrame(summary).to_csv(ROOT / "results/R8_real_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "results/R8_real_patient_scores.csv", index=False)
    (ROOT / "results/R8_real_run.json").write_text(
        json.dumps({"plugin_probability_mass_max_error": float(maxmass)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    started = datetime.now(timezone.utc).isoformat()
    simulation()
    real()
    (ROOT / "results/R8_execution.json").write_text(
        json.dumps(
            {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
        + "\n"
    )
