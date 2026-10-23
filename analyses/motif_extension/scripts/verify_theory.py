"""Independent exact checks of the bivariate-sufficiency characterization.

Uses integer/fraction enumeration for the central examples, independent
linear-program maximum flows for random capacity inequalities, and SciPy
hypergeometric probabilities to validate the C++ bivariate integrator.
"""

from pathlib import Path
from fractions import Fraction as Q
from itertools import product, combinations
import json, math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import multivariate_hypergeom
from pair_flows import integrate_pairs, multinomial_pairs

ROOT = Path(__file__).resolve().parents[1]


def comps(n, d):
    if d == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for tail in comps(n - a, d - 1):
            yield (a,) + tail


def f_lp(x, edges):
    if len(edges) == 0:
        return 0.0
    A = np.zeros((len(x), len(edges)))
    for j, (u, v) in enumerate(edges):
        A[u, j] = A[v, j] = 1
    z = linprog(-np.ones(len(edges)), A_ub=A, b_ub=x, bounds=(0, None), method="highs")
    assert z.success
    return -z.fun


def mean_wedge(p, k):
    """Exact rational flow for central u and partners v,w plus remainder."""
    ans = [Q(0), Q(0), Q(0)]
    pp = list(p) + [1 - sum(p)]
    for m in comps(k, 4):
        w = Q(math.factorial(k), math.prod(math.factorial(t) for t in m))
        for a, b in zip(pp, m):
            w *= a**b
        fs = (min(m[0], m[1] + m[2]), min(m[0], m[2]), min(m[0], m[1]))
        for j, f in enumerate(fs):
            ans[j] += w * Q(f, k)
    return ans


def parity_states(s, parity, complements=True):
    out = []
    for bits in product([0, 1], repeat=3):
        if sum(bits) % 2 == parity:
            active = tuple(s * b for b in bits)
            row = active + tuple(s * (1 - b) for b in bits) if complements else active
            out.append(row + (1 - sum(row),))
    return out


def marginal(rows, inds):
    out = {}
    for row in rows:
        key = tuple(row[i] for i in inds)
        out[key] = out.get(key, Q(0)) + Q(1, len(rows))
    return out


def run():
    rng = np.random.default_rng(26101901)
    report = {}
    records = []
    marginal_checks = 0
    flow_checks = 0
    for s in [Q(1, 10000), Q(1, 100), Q(1, 10), Q(1, 4)]:
        states = [parity_states(s, t) for t in [0, 1]]
        for j in range(7):
            assert marginal(states[0], [j]) == marginal(states[1], [j])
            marginal_checks += 1
        for ij in combinations(range(7), 2):
            assert marginal(states[0], ij) == marginal(states[1], ij)
            marginal_checks += 1
        assert all(sum(row[:-1]) == 3 * s for pop in states for row in pop)
        for k in range(2, 9):
            A = []
            for pop in states:
                fs = [mean_wedge(row[:3], k) for row in pop]
                A.append([sum(f[j] for f in fs) / 4 for j in range(3)])
            rs = [min(f[1:]) / f[0] for f in A]
            assert rs[0] == Q(1, 2)
            assert (rs[1] == Q(1, 2)) if k == 2 else (rs[1] > Q(1, 2))
            if k == 3:
                assert A[0][0] == s * s - s * s * s
                assert A[1][0] == s * s - Q(3, 2) * s**3
                assert rs[1] == (1 - s) / (2 - 3 * s)
            flow_checks += 1
            records.append(
                dict(
                    s=str(s),
                    k=k,
                    R_even_exact=str(rs[0]),
                    R_odd_exact=str(rs[1]),
                    R_odd=float(rs[1]),
                )
            )
    report["exact_pairwise_marginal_checks"] = marginal_checks
    report["exact_parity_resolution_checks"] = flow_checks
    # Every labeled 3-by-3 bipartite graph: matching or contains a wedge.
    allE = list(product(range(3), range(3, 6)))
    matching = 0
    nonmatching = 0
    for mask in range(1, 2**9):
        E = [e for j, e in enumerate(allE) if mask >> j & 1]
        deg = np.bincount(np.array(E).ravel(), minlength=6)
        if deg.max() <= 1:
            matching += 1
            x = rng.integers(0, 7, 6)
            assert abs(f_lp(x, E) - sum(min(x[u], x[v]) for u, v in E)) < 1e-9
        else:
            nonmatching += 1
            v = int(np.argmax(deg))
            nb = [b if a == v else a for a, b in E if v in (a, b)]
            u, w = nb[:2]
            for bits in product([0, 1], repeat=3):
                x = np.zeros(6)
                x[[v, u, w]] = bits
                assert f_lp(x, E) == min(bits[0], bits[1] + bits[2])
    report["all_labeled_3_by_3_nonempty_graphs"] = dict(
        matching=matching, with_wedge=nonmatching, total=matching + nonmatching
    )
    # Competition loss upper bound: sum isolated edge flows minus full flow
    # <= sum of the minima on all graph wedges, <= sum of their products
    # for integer counts. Also removal compensation is nonnegative.
    capacity_checks = 0
    for t in range(180):
        da = int(rng.integers(2, 6))
        db = int(rng.integers(2, 6))
        d = da + db
        E = [(u, v) for u in range(da) for v in range(da, d) if rng.random() < 0.45]
        if not E:
            continue
        x = rng.integers(0, 10, d)
        F = f_lp(x, E)
        b = np.array([min(x[u], x[v]) for u, v in E])
        loss = b.sum() - F
        adjacency = [[w if v == a else v for v, w in E if a in (v, w)] for a in range(d)]
        wedges = [(a, u, v) for a, nb in enumerate(adjacency) for u, v in combinations(nb, 2)]
        W = sum(min(x[a], x[u], x[v]) for a, u, v in wedges)
        assert -1e-9 <= loss <= W + 1e-9
        assert W <= sum(x[a] * x[u] * x[v] for a, u, v in wedges)
        for j, e in enumerate(E):
            deleted = f_lp(x, E[:j] + E[j + 1 :])
            comp = b[j] - (F - deleted)
            assert -1e-9 <= comp <= loss + 1e-9
            capacity_checks += 1
    report["independent_LP_deletion_compensation_checks"] = capacity_checks
    # Independent probability enumeration: ordinary depths and L=k boundaries.
    X = rng.integers(0, 15, (90, 2))
    L = X.sum(1) + rng.integers(0, 120, 90)
    L = np.maximum(L, 2)
    errors = []
    for i, (x, ell) in enumerate(zip(X, L)):
        for k in sorted(set([2, min(8, int(ell)), int(ell)])):
            got = integrate_pairs(x[None, :], np.array([ell]), np.array([[0, 1]]), [k])[0, 0, 0]
            ref = 0.0
            for a in range(1, min(int(x[0]), k) + 1):
                for b in range(1, min(int(x[1]), k - a) + 1):
                    ref += (
                        multivariate_hypergeom.pmf(
                            [a, b, k - a - b], [x[0], x[1], ell - x.sum()], k
                        )
                        * min(a, b)
                        / k
                    )
            errors.append(abs(got - ref))
            assert abs(got - ref) < 2e-12, (x, ell, k, got, ref)
    report["bivariate_probability_checks"] = len(errors)
    report["bivariate_max_absolute_error"] = max(errors)
    # The multinomial comparator is checked independently by complete sums,
    # including two very sparse high-k examples and the p+q=1 boundary.
    from scipy.stats import multinomial

    plugin_errors = []
    for p, k in [
        ([0.03, 0.05], 8),
        ([0.21, 0.37], 32),
        ([0.2, 0.8], 20),
        ([1e-5, 2e-5], 500),
        ([0.0003, 0.001], 500),
    ]:
        ref = 0.0
        for a in range(1, k):
            bs = np.arange(1, k - a + 1)
            mm = np.column_stack([np.full(len(bs), a), bs, k - a - bs])
            ref += np.sum(multinomial.pmf(mm, k, [p[0], p[1], 1 - sum(p)]) * np.minimum(a, bs) / k)
        got = multinomial_pairs(np.array([p]), k)[0]
        plugin_errors.append(abs(got - ref))
        assert abs(got - ref) < 2e-12
    report["multinomial_complete_enumeration_checks"] = len(plugin_errors)
    report["multinomial_max_absolute_error"] = max(plugin_errors)
    report["all_passed"] = True
    (ROOT / "results/theory_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "results/parity_exact_values.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
