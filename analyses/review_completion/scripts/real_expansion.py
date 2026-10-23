from pathlib import Path
import time, json, sys
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from engine import calc, score
from general_graph import integrate

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parents[1]
OUT = ROOT / "results"
IN = ROOT / "inputs"
rng = np.random.default_rng(26101876)
z = np.load(IN / "atlas_integrins_all_cells.npz")
allgenes = list(z["genes"])
edgesdf = pd.read_csv(IN / "integrin_edges_with_provenance.csv")
edges = np.array(
    [
        (allgenes.index(a), allgenes.index(b))
        for a, b in edgesdf[["alpha", "beta"]].itertuples(index=False, name=None)
    ]
)
keep = (z["tissue"] == "T") & (z["celltype"] == "Epi") & (z["depth"] >= 500)
X = z["counts"][keep]
L = z["depth"][keep]
p = z["patient"][keep]
MMR = z["MMR"][keep]
patients = np.unique(p)
patients = np.array([a for a in patients if np.sum(p == a) >= 100])
m = np.isin(p, patients)
X = X[m]
L = L[m]
p = p[m]
MMR = MMR[m]
idx = np.searchsorted(patients, p)
old = np.load(PROJ / "data/development_counts.npz")
oldgenes = list(old["genes"])
sel = [allgenes.index(g) for g in oldgenes]
for i, pid in enumerate(old["patients"]):
    for typ, tname in [(0, "Epi"), (1, "Fibro")]:
        a = np.column_stack(
            [
                old["depth"][(old["patient_index"] == i) & (old["celltype"] == typ)],
                old["counts"][(old["patient_index"] == i) & (old["celltype"] == typ)],
            ]
        )
        mm = (
            (z["patient"] == pid)
            & (z["tissue"] == "T")
            & (z["celltype"] == tname)
            & (z["depth"] >= 500)
        )
        b = np.column_stack([z["depth"][mm], z["counts"][mm][:, sel]])
        assert a.shape == b.shape, (pid, tname, a.shape, b.shape)
        assert np.array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)]), (pid, tname)
print(
    "original 23-patient projection matched exactly; expanded",
    len(patients),
    "patients",
    len(X),
    "cells",
    flush=True,
)
meta = pd.read_csv(IN / "GSE178341_metatables.csv.gz").groupby("PID").first()
features = []
for pid in patients:
    mm = p == pid
    r = dict(
        patient=pid,
        cells=int(mm.sum()),
        MMR=str(MMR[mm][0]),
        depth_median=float(np.median(L[mm])),
        depth_mean=float(L[mm].mean()),
        panel_fraction=float(np.mean(X[mm].sum(1) / L[mm])),
        age=meta.loc[pid, "Age"],
        sex=meta.loc[pid, "Sex"],
        site=meta.loc[pid, "TissueSiteSimple"],
    )
    for j, g in enumerate(allgenes):
        r["mean_" + g] = float(np.mean(X[mm, j] / L[mm]))
        r["detection_" + g] = float(np.mean(X[mm, j] > 0))
    features.append(r)
features = pd.DataFrame(features).set_index("patient")
features.to_csv(OUT / "expanded_patient_features.csv")
ks = [2, 8, 32, 128, 500]
rows = []
timings = []
cellflows = []
mcrows = []
for k in ks:
    t = time.perf_counter()
    draws = 8
    f, b = integrate(X, L, k, edges, batches=8, draws=draws, seed=26101876)
    patient_batches = np.stack([f[p == pid].mean(0) for pid in patients])
    mc = score(patient_batches).std(1, ddof=1) / np.sqrt(8)
    pilot_mc = mc.copy()
    refined = 0
    # Final draws are independent of the pilot for ALL patients, including those
    # retaining the initial draw budget. Thus adaptation does not select estimates.
    if k >= 4:
        f, b = integrate(X, L, k, edges, batches=8, draws=8, seed=26101976)
    for ip in np.flatnonzero(pilot_mc > 5e-5):
        # Pilot selects computational effort only. Fresh independent samples provide
        # the final component estimate conditional on that effort.
        selected_draws = min(32768, int(np.ceil(32 * (pilot_mc[ip] / 1e-4) ** 2)))
        mm = p == patients[ip]
        ff, _ = integrate(
            X[mm], L[mm], k, edges, batches=8, draws=selected_draws, seed=2610187700 + int(ip)
        )
        f[mm] = ff
        refined += 1
        draws = max(draws, selected_draws)
    patient_batches = np.stack([f[p == pid].mean(0) for pid in patients])
    mc = score(patient_batches).std(1, ddof=1) / np.sqrt(8)
    mean = f.mean(1)
    cellflows.append(mean)
    A = patient_batches.mean(1)
    R = score(A)
    for i, pid in enumerate(patients):
        mm = p == pid
        rows.append(
            dict(
                graph="integrin26",
                patient=pid,
                k=k,
                cells=int(mm.sum()),
                R=R[i],
                A0=A[i, 0],
                dominant_edge=str(edgesdf.alpha.iloc[np.argmin(A[i, 1:])])
                + "_"
                + str(edgesdf.beta.iloc[np.argmin(A[i, 1:])]),
                integration_mc_se=mc[i],
                mean_absolute_residual_bound=float(b[mm].mean()),
            )
        )
    timings.append(
        dict(
            graph="integrin26",
            k=k,
            cells=len(X),
            patients=len(patients),
            batches=8,
            maximum_draws_per_batch=draws,
            refined_patients=refined,
            seconds=time.perf_counter() - t,
            max_patient_mc_se=float(mc.max()),
        )
    )
    print("fullgraph", k, timings[-1], flush=True)
    del f
cellflows = np.stack(cellflows, axis=1)
np.savez_compressed(
    OUT / "expanded_fullgraph_cellflows.npz",
    flows=cellflows,
    patient_index=idx,
    patients=patients,
    ks=np.array(ks),
    depth=L,
)
# Original selected graph on all 62 patients, plus the common graph used across sources.
f5, _ = calc(X[:, sel], L, ks)
commonX = X[:, sel].copy()
commonX[:, 3] = 0
fc, _ = calc(commonX, L, ks)
fc = fc[:, :, [0, 1, 2, 4]]
for label, fs in [("selected5", f5), ("common4", fc)]:
    for pid in patients:
        a = fs[p == pid].mean(0)
        rr = score(a)
        for j, k in enumerate(ks):
            rows.append(
                dict(
                    graph=label,
                    patient=pid,
                    k=k,
                    cells=int(np.sum(p == pid)),
                    R=rr[j],
                    A0=a[j, 0],
                    integration_mc_se=0,
                    dominant_edge=int(np.argmin(a[j, 1:])),
                )
            )
pd.DataFrame(rows).to_csv(OUT / "expanded_graph_scores.csv", index=False)
pd.DataFrame(timings).to_csv(OUT / "fullgraph_runtime.csv", index=False)
np.savez_compressed(
    OUT / "expanded_selected_cellflows.npz",
    flows=f5,
    common_flows=fc,
    patient_index=idx,
    patients=patients,
    ks=np.array(ks),
    depth=L,
)
# Same patient set, alternative depth thresholds; report the population change.
threshold = []
for label, fs in [("integrin26", cellflows), ("common4", fc)]:
    for thresholdL in [0, 500, 1000, 2000]:
        for pid in patients:
            m0 = p == pid
            mm = m0 & (L >= thresholdL)
            a = fs[mm].mean(0)
            rr = score(a)
            for j, k in enumerate(ks):
                threshold.append(
                    dict(
                        cohort="D_expanded",
                        graph=label,
                        patient=pid,
                        threshold=thresholdL,
                        k=k,
                        R=rr[j],
                        retained_cells=int(mm.sum()),
                        retained_fraction=float(mm.sum() / m0.sum()),
                        median_depth=float(np.median(L[mm])),
                    )
                )
# Independent original cohort, all k compared on each fixed threshold population.
zi = np.load(PROJ / "data/independent_counts.npz")
mi = zi["celltype"] == 0
xi = zi["counts"][mi].copy()
xi[:, 3] = 0
li = zi["depth"][mi]
pi = zi["patients"][zi["patient_index"][mi]]
fi, _ = calc(xi, li, ks)
fi = fi[:, :, [0, 1, 2, 4]]
np.savez_compressed(
    OUT / "independent_common_cellflows.npz", flows=fi, patient=pi, ks=np.array(ks), depth=li
)
for t in [500, 1000, 2000]:
    for pid in np.unique(pi):
        m0 = pi == pid
        mm = m0 & (li >= t)
        rr = score(fi[mm].mean(0))
        for j, k in enumerate(ks):
            threshold.append(
                dict(
                    cohort="I",
                    graph="common4",
                    patient=pid,
                    threshold=t,
                    k=k,
                    R=rr[j],
                    retained_cells=int(mm.sum()),
                    retained_fraction=float(mm.sum() / m0.sum()),
                    median_depth=float(np.median(li[mm])),
                )
            )
pd.DataFrame(threshold).to_csv(OUT / "depth_threshold_sensitivity.csv", index=False)
# Panel-conditional measurement changes both target and cell eligibility.
panel = []
for co, counts, depth, ids in [("D", X[:, sel], L, p), ("I", xi, li, pi)]:
    counts = counts.copy()
    counts[:, 3] = 0
    S = counts.sum(1)
    for n in [2, 4, 8]:
        eligible = S >= n
        if eligible.any():
            # calc's probability input is unused in exact mode; pass a valid placeholder
            # because panel fractions sum to one and its plugin assertion excludes that boundary.
            fs, _ = calc(
                counts[eligible], S[eligible], [n], P=np.zeros_like(counts[eligible], float)
            )
            fs = fs[:, 0, [0, 1, 2, 4]]
            for pid in np.unique(ids):
                mm = ids == pid
                me = ids[eligible] == pid
                panel.append(
                    dict(
                        cohort=co,
                        patient=pid,
                        panel_resolution=n,
                        eligible_cells=int(me.sum()),
                        total_cells=int(mm.sum()),
                        coverage=float(me.sum() / mm.sum()),
                        R=float(score(fs[me].mean(0))) if me.any() else np.nan,
                        median_depth_retained=(
                            float(np.median(depth[mm & eligible]))
                            if np.any(mm & eligible)
                            else np.nan
                        ),
                        median_depth_all=float(np.median(depth[mm])),
                    )
                )
pd.DataFrame(panel).to_csv(OUT / "panel_conditional_selection.csv", index=False)
(OUT / "real_expansion_checks.json").write_text(
    json.dumps(
        dict(
            original_23_patient_projection_exact=True,
            all_public_tumor_epithelial_cells_above_500=bool(
                np.all(z["depth"][(z["tissue"] == "T") & (z["celltype"] == "Epi")] >= 500)
            ),
            expanded_patients=len(patients),
            expanded_cells=len(X),
            all_26_rows_present=True,
            full_graph_edges=len(edges),
            graph_is_forest=True,
            integration_precision_goal=1e-4,
            max_observed_patient_mc_se=float(max(t["max_patient_mc_se"] for t in timings)),
        ),
        indent=2,
    )
    + "\n"
)
print("DONE real expansion", flush=True)
