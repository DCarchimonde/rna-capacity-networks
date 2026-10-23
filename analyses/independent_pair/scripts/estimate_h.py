"""Equal-cell RNA pair estimators, distinct-patient ranking metrics."""

import numpy as np
from scipy.stats import rankdata

EDGES = [(0, 2), (1, 2), (1, 3), (1, 4)]
EDGE_NAMES = ["ITGA5_ITGB1", "ITGAV_ITGB1", "ITGAV_ITGB3", "ITGAV_ITGB6"]
METHODS = ["cell_factorial", "cell_fraction_product", "crosscell_marginal"]


def type_scores(x, l):
    if len(l) < 2:
        return np.full((3, 4), np.nan)
    x, l = np.asarray(x, dtype=float), np.asarray(l, dtype=float)
    if np.any(l < 2):
        raise ValueError("H requires a denominator of at least two UMIs")
    y = x / l[:, None]
    prod = np.stack([y[:, i] * y[:, j] for i, j in EDGES], axis=1)
    h = prod * (l / (l - 1))[:, None]
    n = len(l)
    sums = y.sum(axis=0)
    u = np.array(
        [(sums[i] * sums[j] - prod[:, e].sum()) / (n * (n - 1)) for e, (i, j) in enumerate(EDGES)]
    )
    return np.stack([h.mean(axis=0), prod.mean(axis=0), u])


def cohort_scores(x, l, p, t, n):
    a = np.full((n, 3, 3, 4), np.nan)
    for patient in range(n):
        for typ in range(2):
            sel = (p == patient) & (t == typ)
            a[patient, typ] = type_scores(x[sel], l[sel])
        a[patient, 2] = (a[patient, 0] + a[patient, 1]) / 2
    return a


def pair_metrics(ref, query, original_ids=None):
    ref, query = np.asarray(ref), np.asarray(query)
    n = len(ref)
    i, j = np.triu_indices(n, 1)
    distinct = (
        np.ones(len(i), dtype=bool) if original_ids is None else original_ids[i] != original_ids[j]
    )
    den = int(distinct.sum())
    if den == 0:
        return {
            "all_pair_retention": np.nan,
            "reference_resolution": np.nan,
            "strict_accuracy": np.nan,
            "D": np.nan,
            "distinct_pairs": 0,
            "finite_patients": 0,
        }
    good = distinct & np.isfinite(ref[i]) & np.isfinite(ref[j]) & (ref[i] != ref[j])
    qgood = np.isfinite(query[i]) & np.isfinite(query[j])
    retained = good & qgood & ((ref[i] - ref[j]) * (query[i] - query[j]) > 0)
    valid = np.isfinite(ref) & np.isfinite(query)
    # D is reported only for original cohorts, where ranks have no bootstrap copies.
    d = np.nan
    if original_ids is None and valid.sum() > 1:
        r = rankdata(ref[valid], method="average")
        q = rankdata(query[valid], method="average")
        d = np.abs(r - q).sum() / (valid.sum() * (valid.sum() - 1))
    return {
        "all_pair_retention": float(retained.sum() / den),
        "reference_resolution": float(good.sum() / den),
        "strict_accuracy": float(retained.sum() / good.sum()) if good.any() else np.nan,
        "D": float(d),
        "distinct_pairs": den,
        "finite_patients": int(valid.sum()),
    }
