"""Only resolve directions exceeding a floating-point arithmetic bound."""

import numpy as np
from scipy.stats import rankdata


def difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    tol = 64 * np.finfo(float).eps * np.maximum(abs(a), abs(b))
    return np.where(abs(d) <= tol, 0.0, d)


def pair_metrics(ref, query):
    ref = np.asarray(ref, float)
    query = np.asarray(query, float)
    n = len(ref)
    i, j = np.triu_indices(n, 1)
    dr, dq = difference(ref[i], ref[j]), difference(query[i], query[j])
    good = np.isfinite(dr) & (dr != 0)
    qgood = np.isfinite(dq)
    kept = good & qgood & (dr * dq > 0)
    reversed_ = good & qgood & (dr * dq < 0)
    finite = np.isfinite(ref) & np.isfinite(query)
    if finite.sum() > 1:
        d = np.abs(rankdata(ref[finite]) - rankdata(query[finite])).sum() / (
            finite.sum() * (finite.sum() - 1)
        )
    else:
        d = np.nan
    return {
        "D": float(d),
        "reference_resolution": float(good.mean()),
        "query_resolution": float((qgood & (dq != 0)).mean()),
        "strict_accuracy": float(kept.sum() / good.sum()) if good.any() else np.nan,
        "all_pair_retention": float(kept.mean()),
        "reversal_fraction": float(reversed_.mean()),
        "finite_patients": int(finite.sum()),
        "strict_reference_pairs": int(good.sum()),
    }
