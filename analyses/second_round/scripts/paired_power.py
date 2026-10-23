"""Paired permutation comparison: two populations x two statistics.

Regenerate the original cohorts, keeping the original evaluation sets. Both
statistics use identical conditional component estimates and the same 199 label
permutations within a replicate. This is a new, fixed-seed permutation analysis,
not an exact replay of the previous permutation draws or calibrated R test.
"""

from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "analyses/motif_extension/scripts"))
sys.path.insert(0, str(PROJECT / "analyses/review_completion/scripts"))
from parity_power import generate, truth as parity_truth
from pair_flows import integrate_pairs, multinomial_pairs
from engine import calc, population

OUT = ROOT / "results"
CACHE = ROOT / "cache"
S, L, N, REPS, K, PERM = 1e-4, 10000, 3000, 256, 500, 199
PERM_SEED = 2610192100
EDGES = np.array([[0, 2], [1, 2], [1, 3], [1, 4]])


def cohort_cache():
    path = CACHE / "original_evaluation_cohorts.npz"
    if path.exists():
        return {k: v for k, v in np.load(path).items()}
    result = {}
    rng = np.random.default_rng(26101903)
    for s in [1e-5, 1e-4]:
        for depth in [1000, 10000]:
            for setting, a, b in [("alternative", 0.5, -0.5), ("null", 0.0, 0.0)]:
                x = generate(rng, s, depth, a, REPS * N)
                y = generate(rng, s, depth, b, REPS * N)
                if s == S and depth == L:
                    result[f"parity_{setting}"] = (
                        np.stack([x, y]).reshape(2, REPS, N, 3).astype(np.int16)
                    )
    # Exact generation order of detection_power.py at its first scale. Only the
    # second 256 replicates are the original evaluation sets; the first 256
    # supplied independent calibration data and are not used in the new tests.
    rng = np.random.default_rng(26101879)
    r = 512
    j = np.empty((3, r, N), int)
    j[:2] = 2 * (rng.random((2, r, N)) < 0.75)
    j[2] = 1 + 2 * (rng.random((r, N)) < 0.25)
    p = np.full((3 * r * N, 5), S)
    p[:, 0] = S * j.reshape(-1)
    remaining = np.full(len(p), L, int)
    probability = np.ones(len(p))
    x = np.empty((len(p), 5), np.int64)
    for g in range(5):
        x[:, g] = rng.binomial(remaining, p[:, g] / probability)
        remaining -= x[:, g]
        probability -= p[:, g]
    x = x.reshape(3, r, N, 5)[:, 256:].astype(np.int16)
    result["matched_null"] = x[[0, 1]]
    result["matched_alternative"] = x[[0, 2]]
    np.savez_compressed(path, **result)
    return result


def map_components(x, family):
    """Columns: per-cell D, intact A, each separate deletion component.

    R is formed AFTER group averaging; averaging cellwise minima would target
    a different quantity and is not used anywhere in this comparison.
    """
    unique, inverse = np.unique(x.reshape(-1, x.shape[-1]), axis=0, return_inverse=True)
    if family == "parity":
        pairs = np.concatenate(
            [
                unique[:, [0, 1]],
                unique[:, [0, 2]],
                np.column_stack([unique[:, 0], unique[:, 1] + unique[:, 2]]),
            ]
        )
        pu, pi = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True)
        h = integrate_pairs(pu, np.full(len(pu), L), np.array([[0, 1]]), [K])[:, 0, 0]
        b1, b2, intact = h[pi].reshape(3, len(unique))
        loss = b1 + b2 - intact
        loss[np.any(unique == 0, axis=1)] = 0.0
        values = np.column_stack([loss, intact, b1, b2])
    else:
        f, _ = calc(unique, L, [K])
        f = f[:, 0]
        b = integrate_pairs(unique, np.full(len(unique), L), EDGES, [K])[:, 0].sum(1)
        values = np.column_stack([b - f[:, 0], f])
        values[(unique > 0).sum(1) <= 2, 0] = 0.0
    assert values[:, 0].min() > -1e-12
    values[:, 0] = np.maximum(values[:, 0], 0.0)
    # Floating-point roundoff must not change the permutation category order.
    # Integer labels encode the original partition; values are recalculated.
    with np.load(ROOT / "inputs/permutation_groups.npz") as partition:
        assert np.array_equal(unique, partition[family + "_counts"])
        reference = partition[family + "_reference_components"]
        tolerance = 64 * np.finfo(float).eps * max(float(np.abs(reference).max()), 1e-300)
        assert np.allclose(values, reference, rtol=0, atol=tolerance)
        labels = partition[family + "_groups"]
        _, first, category = np.unique(labels, return_index=True, return_inverse=True)
        canonical = values[first][category]
        assert np.allclose(values, canonical, rtol=0, atol=tolerance)
        # The observed and permuted statistics use the same numerical identities.
        values = canonical
        groups = labels[inverse].reshape(x.shape[:-1])
    np.savez_compressed(CACHE / f"{family}_component_lookup.npz", counts=unique, components=values)
    return values[inverse].reshape(*x.shape[:-1], values.shape[-1]), groups


def contrast(a, b):
    d = np.abs(a[..., 0] - b[..., 0]) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(a[..., 2:].min(-1) / a[..., 1] - b[..., 2:].min(-1) / b[..., 1])
    return np.stack([d, r], axis=-1)


def truth_table():
    rows = []
    for k in [3, 4, 5, 500]:
        for theta in [0.5, -0.5]:
            v = parity_truth(S, theta, k)
            rows.append(
                dict(
                    family="parity",
                    group=0 if theta > 0 else 1,
                    k=k,
                    D=v["loss"],
                    R=v["R"],
                    A=v["A"],
                    B=v["B"],
                )
            )
        f, _ = population(S, [k])
        p = S * np.array([[j, 1, 1, 1, 1] for j in range(4)])
        b = sum(multinomial_pairs(p[:, [u, v]], k) for u, v in EDGES)
        for group, w in enumerate([np.array([0.25, 0, 0.75, 0]), np.array([0, 0.75, 0, 0.25])]):
            a = f[group, 0, 0]
            rows.append(
                dict(
                    family="matched",
                    group=group,
                    k=k,
                    D=w @ b - a,
                    R=f[group, 0, 1:].min() / a,
                    A=a,
                    B=w @ b,
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "population_truth.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    spec = dict(
        s=S,
        depth=L,
        cells_per_group=N,
        repetitions=REPS,
        k=K,
        permutations=PERM,
        parity_generation_seed=26101903,
        matched_generation_seed=26101879,
        permutation_seed=PERM_SEED,
        order=["parity_alternative", "parity_null", "matched_alternative", "matched_null"],
        statistics=[
            "absolute group-mean D difference",
            "absolute R difference; min after component means",
        ],
        estimator="conditional hypergeometric for both statistics",
        pvalue="(1 + number of permuted statistics >= observed, including numerical ties)/200; reject <= .05",
        null="identical cell-count distributions; exchangeable labels, not merely equality of R or D",
        zero_baseline="if observed or permuted R undefined, count test as non-rejected and report frequency",
        pairing="same counts and permutations for D and R within each replicate",
        category_partition="Fixed integer labels from the original component lookup; recalculated values are checked and canonicalised within 64 machine epsilons of the component scale.",
        provenance="original evaluation cohorts regenerated exactly; permutation draws newly fixed before inspecting results",
        scope="descriptive crossed comparison, not a universal causal attribution of power to moment order",
    )
    (ROOT / "paired_power_design.json").write_text(json.dumps(spec, indent=2) + "\n")
    tic = time.perf_counter()
    cohorts = cohort_cache()
    rng = np.random.default_rng(PERM_SEED)
    records, summaries = [], []
    for family in ["parity", "matched"]:
        # One shared lookup per family; retain the original alternative/null pairs.
        raw = np.concatenate([cohorts[f"{family}_alternative"], cohorts[f"{family}_null"]], axis=0)
        mapped, group_ids = map_components(raw, family)
        for setting, indices in [("alternative", (0, 1)), ("null", (2, 3))]:
            all_p, all_valid = [], []
            for rep in range(REPS):
                av, bv = mapped[indices[0], rep], mapped[indices[1], rep]
                pooled = np.concatenate([av, bv])
                pooled_groups = np.concatenate(
                    [group_ids[indices[0], rep], group_ids[indices[1], rep]]
                )
                _, first, counts = np.unique(pooled_groups, return_index=True, return_counts=True)
                unique = pooled[first]
                obs = contrast(av.sum(0), bv.sum(0))
                allocation = rng.multivariate_hypergeometric(counts, N, size=PERM)
                total = counts @ unique
                sum0 = allocation @ unique
                stats = contrast(sum0, total - sum0)
                valid = np.isfinite(obs) & np.isfinite(stats).all(0)
                tol = (
                    64
                    * np.finfo(float).eps
                    * np.maximum(np.maximum(np.nanmax(stats, axis=0), obs), 1e-300)
                )
                ps = np.where(valid, (1 + (stats >= obs - tol).sum(0)) / (PERM + 1), 1.0)
                all_p.append(ps)
                all_valid.append(valid)
                for j, name in enumerate(["D", "R500"]):
                    records.append(
                        dict(
                            family=family,
                            setting=setting,
                            replicate=rep,
                            statistic=name,
                            observed=obs[j],
                            pvalue=ps[j],
                            valid=bool(valid[j]),
                            reject=bool(ps[j] <= 0.05),
                        )
                    )
            all_p, all_valid = np.array(all_p), np.array(all_valid)
            for j, name in enumerate(["D", "R500"]):
                hits = int((all_p[:, j] <= 0.05).sum())
                ci = binomtest(hits, REPS).proportion_ci()
                summaries.append(
                    dict(
                        family=family,
                        setting=setting,
                        statistic=name,
                        rejections=hits,
                        repetitions=REPS,
                        rejection_rate=hits / REPS,
                        ci_low=ci.low,
                        ci_high=ci.high,
                        undefined_tests=int((~all_valid[:, j]).sum()),
                        permutations=PERM,
                    )
                )
            print(f"{family} {setting}: {summaries[-2:]}", flush=True)
            pd.DataFrame(records).to_csv(OUT / "paired_permutation_replicates.csv", index=False)
            pd.DataFrame(summaries).to_csv(OUT / "paired_permutation_summary.csv", index=False)
    truth_table()
    inputs = [
        CACHE / "original_evaluation_cohorts.npz",
        Path(__file__),
        PROJECT / "analyses/motif_extension/scripts/parity_permutation.py",
        PROJECT / "analyses/review_completion/scripts/detection_power.py",
    ]
    (OUT / "run_record.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - tic,
                numpy=np.__version__,
                hashes={
                    str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in inputs
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Regenerate evaluation counts and component lookups from fixed seeds.",
    )
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    if args.rebuild_cache:
        for name in [
            "original_evaluation_cohorts.npz",
            "matched_component_lookup.npz",
            "parity_component_lookup.npz",
        ]:
            (CACHE / name).unlink(missing_ok=True)
    main()
