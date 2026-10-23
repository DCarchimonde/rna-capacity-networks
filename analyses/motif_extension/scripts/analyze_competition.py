"""Decompose a fixed-k score into bivariate marginal and shared-capacity effects.

This reuses previously exposed cohorts. No outcome is used to choose a
resolution, graph, or approximation. Full-graph conditional flows use the final numerical refinement;
the per-edge comparator is
exact conditional integration. All differences here concern the same k and
same observed cells, but independent-edge and capacity-constrained graphs
are different mechanistic targets, not rival estimators of the same target.
"""

from pathlib import Path
from itertools import combinations
import argparse, json, time
import numpy as np
import pandas as pd
from pair_flows import integrate_pairs, wedge_moments

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def score(A):
    return np.min(A[..., 1:], axis=-1) / A[..., 0]


def pairscore(B):
    return 1 - np.max(B, axis=-1) / np.sum(B, axis=-1)


def count_inversions(a, b, tol=1e-12):
    count = ties = 0
    for i, j in combinations(range(len(a)), 2):
        x = a[i] - a[j]
        y = b[i] - b[j]
        if abs(x) <= tol or abs(y) <= tol:
            ties += 1
        elif x * y < 0:
            count += 1
    return count, ties


def summarize(label, X, L, p, patients, genes, edges, ks, F):
    t = time.perf_counter()
    B = integrate_pairs(X, L, edges, ks)
    elapsed = time.perf_counter() - t
    print(label, "exact pair integration seconds", elapsed, flush=True)
    wedges, W = wedge_moments(X, L, edges)
    np.savez_compressed(
        OUT / (label + "_bivariate_cellflows.npz"), flows=B, patients=patients, patient=p, ks=ks
    )
    rows = []
    edge_rows = []
    motif_rows = []
    rng = np.random.default_rng(26101902)
    for pid in patients:
        take = p == pid
        n = int(take.sum())
        bb = B[take]
        ff = F[take]
        ww = W[take]
        AB = bb.mean(0)
        AF = ff.mean(0)
        wbar = ww.sum(1).mean()
        rp = pairscore(AB)
        rf = score(AF)
        assert abs(rp[0] - rf[0]) < 1e-10
        r2 = rf[0]
        bsum = AB.sum(1)
        c = bsum - AF[:, 0]
        cbar = (ks - 1) * (ks - 2) * wbar
        # Smooth competition fraction only: no minimum/edge selection in this CI.
        y = np.column_stack([bb[:, -1].sum(1), ff[:, -1, 0]])
        boot = []
        for start in range(0, 512, 32):
            ix = rng.integers(0, n, size=(32, n))
            means = y[ix].mean(1)
            values = np.full(len(means), np.nan)
            positive = means[:, 0] > 0
            values[positive] = 1 - means[positive, 1] / means[positive, 0]
            boot.extend(values.tolist())
        undefined_bootstrap = int(np.sum(~np.isfinite(boot)))
        ci = (
            np.quantile(boot, [0.025, 0.975])
            if undefined_bootstrap == 0
            else np.array([np.nan, np.nan])
        )
        for h, k in enumerate(ks):
            if bsum[h] > cbar[h]:
                lo = max(0.0, 1 - AB[h].max() / (bsum[h] - cbar[h]))
                hi = min(1.0, (bsum[h] - AB[h].max()) / (bsum[h] - cbar[h]))
            else:
                lo = 0.0
                hi = 1.0
            rows.append(
                dict(
                    cohort=label,
                    patient=str(pid),
                    cells=n,
                    k=int(k),
                    R2=r2,
                    R_full=rf[h],
                    R_bivariate=rp[h],
                    total_resolution_change=rf[h] - r2,
                    bivariate_resolution_change=rp[h] - r2,
                    shared_capacity_change=rf[h] - rp[h],
                    B_sum=bsum[h],
                    A0=AF[h, 0],
                    competition_loss=c[h],
                    competition_fraction=c[h] / bsum[h],
                    competition_fraction_ci_low=float(ci[0]) if k == ks[-1] else np.nan,
                    competition_fraction_ci_high=float(ci[1]) if k == ks[-1] else np.nan,
                    competition_fraction_bootstrap_undefined=(
                        undefined_bootstrap if k == ks[-1] else 0
                    ),
                    wedge_upper_bound=cbar[h],
                    R_motif_bound_low=lo,
                    R_motif_bound_high=hi,
                    wedge_support_cells=int(np.sum(ww.sum(1) > 0)),
                    panel_fraction=float(np.mean(X[take].sum(1) / L[take])),
                )
            )
            for j, (u, v) in enumerate(edges):
                actual_loss = AF[h, 0] - AF[h, j + 1]
                edge_rows.append(
                    dict(
                        cohort=label,
                        patient=str(pid),
                        k=int(k),
                        edge=genes[u] + "_" + genes[v],
                        B_edge=AB[h, j],
                        actual_deletion_loss=actual_loss,
                        compensation=AB[h, j] - actual_loss,
                    )
                )
        for j, (u, v, w) in enumerate(wedges):
            motif_rows.append(
                dict(
                    cohort=label,
                    patient=str(pid),
                    leaf1=genes[u],
                    center=genes[v],
                    leaf2=genes[w],
                    factorial_moment=ww[:, j].mean(),
                    support_cells=int(np.sum(ww[:, j] > 0)),
                )
            )
    df = pd.DataFrame(rows)
    pd.DataFrame(edge_rows).to_csv(OUT / (label + "_edge_compensation.csv"), index=False)
    pd.DataFrame(motif_rows).to_csv(OUT / (label + "_wedge_moments.csv"), index=False)
    df.to_csv(OUT / (label + "_competition_decomposition.csv"), index=False)
    summaries = []
    for k in ks:
        d = df[df.k == k]
        a = d.R2.to_numpy()
        b = d.R_bivariate.to_numpy()
        c = d.R_full.to_numpy()
        n_total, _ = count_inversions(a, c)
        n_pair, _ = count_inversions(a, b)
        n_comp, _ = count_inversions(b, c)
        follows_pair = requires_comp = unresolved = 0
        for i, j in combinations(range(len(a)), 2):
            x = a[i] - a[j]
            y = b[i] - b[j]
            z = c[i] - c[j]
            if x * z >= 0:
                continue
            if abs(y) <= 1e-12:
                unresolved += 1
            elif y * z > 0:
                follows_pair += 1
            else:
                requires_comp += 1
        summaries.append(
            dict(
                cohort=label,
                k=int(k),
                patients=len(patients),
                cells=len(X),
                pairs=len(patients) * (len(patients) - 1) // 2,
                inversion_R2_full=n_total,
                inversion_R2_bivariate=n_pair,
                inversion_bivariate_full=n_comp,
                total_inversions_already_in_bivariate=follows_pair,
                total_inversions_requiring_shared_capacity=requires_comp,
                attribution_ties=unresolved,
                median_competition_fraction=float(d.competition_fraction.median()),
                min_competition_fraction=float(d.competition_fraction.min()),
                max_competition_fraction=float(d.competition_fraction.max()),
                median_absolute_shared_score_change=float(d.shared_capacity_change.abs().median()),
                median_absolute_bivariate_score_change=float(
                    d.bivariate_resolution_change.abs().median()
                ),
                min_shared_score_change=float(d.shared_capacity_change.min()),
                max_shared_score_change=float(d.shared_capacity_change.max()),
                median_wedge_bound_width=float(
                    (d.R_motif_bound_high - d.R_motif_bound_low).median()
                ),
                all_decompositions_sum=bool(
                    np.max(
                        abs(
                            d.total_resolution_change
                            - d.bivariate_resolution_change
                            - d.shared_capacity_change
                        )
                    )
                    < 1e-12
                ),
            )
        )
    report = dict(
        cohort=label, pair_integration_seconds=elapsed, wedges=len(wedges), summaries=summaries
    )
    (OUT / (label + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return df, pd.DataFrame(summaries)


def main(project):
    project = Path(project)
    b = project / "analyses/review_completion"
    z = np.load(b / "inputs/atlas_integrins_all_cells.npz")
    genes = list(z["genes"])
    E = pd.read_csv(b / "inputs/integrin_edges_with_provenance.csv")
    edges = np.array(
        [
            (genes.index(u), genes.index(v))
            for u, v in E[["alpha", "beta"]].itertuples(index=False, name=None)
        ]
    )
    ref = np.load(b / "results/expanded_fullgraph_cellflows.npz")
    patients = ref["patients"]
    ks = ref["ks"]
    keep = (
        (z["tissue"] == "T")
        & (z["celltype"] == "Epi")
        & (z["depth"] >= 500)
        & np.isin(z["patient"], patients)
    )
    X = z["counts"][keep]
    L = z["depth"][keep]
    p = z["patient"][keep]
    assert np.array_equal(L, ref["depth"]) and np.array_equal(p, patients[ref["patient_index"]])
    full_flows = ref["flows"].copy()
    refined_path = OUT / "refined_close_cellflows.npz"
    if refined_path.exists():
        refined = np.load(refined_path)
        full_flows[refined["indices"], int(np.flatnonzero(ks == int(refined["k"]))[0])] = refined[
            "flows"
        ]
    d1, s1 = summarize("D_full26", X, L, p, patients, genes, edges, ks, full_flows)
    z = np.load(project / "data/independent_counts.npz")
    genes = list(z["genes"])
    keep = z["celltype"] == 0
    X = z["counts"][keep].copy()
    X[:, 3] = 0
    L = z["depth"][keep]
    p = z["patients"][z["patient_index"][keep]]
    ref = np.load(b / "results/independent_common_cellflows.npz")
    assert np.array_equal(L, ref["depth"]) and np.array_equal(p, ref["patient"])
    E = np.array([[0, 2], [1, 2], [1, 4]])
    d2, s2 = summarize("I_common4", X, L, p, np.unique(p), genes, E, ref["ks"], ref["flows"])
    pd.concat([d1, d2]).to_csv(OUT / "competition_decomposition.csv", index=False)
    pd.concat([s1, s2]).to_csv(OUT / "competition_summary.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(ROOT.parents[1]))
    main(ap.parse_args().project)
