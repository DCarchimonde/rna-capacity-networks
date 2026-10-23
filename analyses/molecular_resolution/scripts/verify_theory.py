"""Exact finite fixtures, independent LP checks, and resolution witnesses."""

from pathlib import Path
from fractions import Fraction as Q
from itertools import combinations
from math import comb, factorial
import json
import numpy as np
from scipy.optimize import linprog
from scipy.special import gammaln
from resolution import flows, conditional_flows, h_flows, r_from_flows

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(0, 2), (1, 2), (1, 3), (1, 4)]


def compositions(n, d):
    if d == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for z in compositions(n - i, d - 1):
                yield (i,) + z


def exact_flow(z, common=False):
    a, v, b, c, d = z
    if common:
        return (
            min(a + v, v + b, b + d),
            min(v, b + d),
            min(a, b) + min(v, d),
            min(a + v, v + b, b),
        )
    return (
        min(a + v, v + b, b + c + d),
        min(v, b + c + d),
        min(a, b) + min(v, c + d),
        min(a + v, v + b, b + d),
        min(a + v, v + b, b + c),
    )


def conditional_exact(x, L, k, common=False):
    out = [Q(0)] * (4 if common else 5)
    mass = Q(0)
    for z in compositions(k, 6):
        cats = (*x, L - sum(x))
        if any(a > b for a, b in zip(z, cats)):
            continue
        w = Q(np.prod([comb(int(a), int(b)) for a, b in zip(cats, z)], dtype=object), comb(L, k))
        mass += w
        for j, f in enumerate(exact_flow(z[:5], common)):
            out[j] += w * Q(f, k)
    assert mass == 1
    return out


def population_exact(pop, k, common=False):
    out = [Q(0)] * (4 if common else 5)
    for z in compositions(k, 6):
        coefficient = Q(factorial(k), np.prod([factorial(a) for a in z], dtype=object))
        for w, p in pop:
            mass = w * coefficient
            for pp, a in zip((*p, 1 - sum(p)), z):
                mass *= pp**a
            for j, f in enumerate(exact_flow(z[:5], common)):
                out[j] += mass * Q(f, k)
    return out


def k3_formula(p, common=False):
    edges = [EDGES[e] for e in ([0, 1, 3] if common else range(4))]
    ans = []
    for deleted in [None] + list(range(len(edges))):
        ee = [e for j, e in enumerate(edges) if j != deleted]
        value = sum(p[a] * p[b] * (2 - p[a] - p[b]) for a, b in ee)
        for left, right in combinations(ee, 2):
            if len(set(left) & set(right)) == 1:
                value -= 2 * np.prod([p[j] for j in set(left) | set(right)])
        ans.append(value)
    return ans


def population_numeric(pop, k, common=False):
    z = np.array(list(compositions(k, 6)), dtype=np.int32)
    logc = gammaln(k + 1) - gammaln(z + 1).sum(1)
    f = flows(z[:, :5] / k, common)
    out = np.zeros(f.shape[1])
    for w, p0 in pop:
        target = np.asarray(p0, float)
        p = np.r_[target, 1 - target.sum()]
        valid = (z[:, p == 0] == 0).all(1)
        positive = p > 0
        weights = np.exp(logc[valid] + z[valid][:, positive] @ np.log(p[positive]))
        out += float(w) * (weights @ f[valid])
    return out


def second_moment_pop(which, s=Q(3, 25)):
    entries = [(0, Q(1, 4)), (2, Q(3, 4))] if which == 0 else [(1, Q(3, 4)), (3, Q(1, 4))]
    return [(w, tuple(s * q for q in [j, 1, 1, 1, 1])) for j, w in entries]


def mean_matched_pop(which, s=Q(1, 50)):
    if which == 0:
        return [
            (Q(1, 4), tuple(s * q for q in z))
            for z in [(4, 0, 4, 0, 0), (0, 4, 0, 0, 0), (0, 0, 0, 4, 0), (0, 0, 0, 0, 4)]
        ]
    return [(Q(1, 2), tuple(s * q for q in z)) for z in [(0, 2, 2, 2, 2), (2, 0, 0, 0, 0)]]


def main():
    checks = 0
    max_error = 0.0
    mass_error = 0.0
    fixtures = [
        ([1, 1, 1, 1, 1], 7),
        ([3, 2, 1, 0, 2], 10),
        ([0, 3, 2, 1, 0], 8),
        ([4, 1, 3, 2, 1], 12),
        ([1, 0, 0, 0, 0], 4),
    ]
    for common in [False, True]:
        for x, L in fixtures:
            for k in range(1, min(L, 6) + 1):
                pred, errs = conditional_flows(np.array([x]), np.array([L]), [k], common)
                truth = conditional_exact(x, L, k, common)
                error = max(abs(float(a) - b) for a, b in zip(truth, pred[0, 0]))
                max_error = max(max_error, error)
                mass_error = max(mass_error, errs.max())
                assert error < 1e-12
                checks += len(truth)
        x = np.array([[3, 2, 1, 0, 2], [1, 1, 1, 1, 1]])
        l = np.array([10, 7])
        pred, _ = conditional_flows(x, l, [2], common)
        assert np.max(np.abs(pred[:, 0] - h_flows(x, l, common))) < 1e-13
    pop = [(Q(1), tuple(Q(v, 20) for v in [1, 2, 3, 1, 2]))]
    unbiased = 0
    for common in [False, True]:
        for k in [2, 3, 4]:
            L = 6
            ans = [Q(0)] * (4 if common else 5)
            p = (*pop[0][1], 1 - sum(pop[0][1]))
            for x in compositions(L, 6):
                mass = Q(factorial(L), np.prod([factorial(a) for a in x], dtype=object))
                for pp, a in zip(p, x):
                    mass *= pp**a
                est = conditional_exact(x[:5], L, k, common)
                for j, v in enumerate(est):
                    ans[j] += mass * v
            truth = population_exact(pop, k, common)
            assert ans == truth
            unbiased += len(truth)
        exact = population_exact(pop, 3, common)
        formula = k3_formula(pop[0][1], common)
        assert exact == formula
    rng = np.random.default_rng(2610105211)
    lp_error = 0.0
    A = np.array([[1, 0, 0, 0], [0, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    for common in [False, True]:
        for p in rng.dirichlet(np.ones(6), size=64)[:, :5]:
            expected = flows(p, common)
            for j, deleted in enumerate([None] + ([0, 1, 3] if common else [0, 1, 2, 3])):
                bounds = [
                    (0, 0) if e == deleted or (common and e == 2) else (0, None) for e in range(4)
                ]
                r = linprog(-np.ones(4), A_ub=A, b_ub=p, bounds=bounds, method="highs")
                assert r.success
                lp_error = max(lp_error, abs(-r.fun - expected[j]))
    assert lp_error < 1e-12
    # Identical full first and second moments: R2 and R3 still coincide,
    # but the graph detects the difference at k=4.
    pa, pb = second_moment_pop(0), second_moment_pop(1)
    witness = []
    for k in [1, 2, 3, 4, 8, 16]:
        if k <= 4:
            aa, bb = population_exact(pa, k), population_exact(pb, k)
            if k in [2, 3]:
                assert aa == bb
            if k == 4:
                assert aa != bb
            Ra = min(aa[1:]) / aa[0] if aa[0] else None
            Rb = min(bb[1:]) / bb[0] if bb[0] else None
            exacta, exactb = str(Ra), str(Rb)
        else:
            aa, bb = population_numeric(pa, k), population_numeric(pb, k)
            Ra, Rb = r_from_flows(aa), r_from_flows(bb)
            exacta = exactb = None
        witness.append(
            {
                "k": k,
                "R_A": float(Ra) if Ra is not None else None,
                "R_B": float(Rb) if Rb is not None else None,
                "R_A_exact": exacta,
                "R_B_exact": exactb,
            }
        )
    # Flow monotonicity follows from concavity. A ratio has no such property.
    ratio_counter = None
    for p in rng.dirichlet(np.ones(6), size=24)[:, :5]:
        aa = np.stack([population_numeric([(1, p)], k) for k in [2, 3, 4, 8]])
        assert (np.diff(aa, axis=0) >= -1e-13).all()
        rr = r_from_flows(aa)
        if (np.diff(rr) > 1e-6).any() and (np.diff(rr) < -1e-6).any():
            ratio_counter = {"p": p.tolist(), "k": [2, 3, 4, 8], "R": rr.tolist()}
            break
    result = {
        "all_passed": True,
        "conditional_fixture_flow_checks": checks,
        "max_abs_conditional_error": max_error,
        "max_hypergeom_mass_error": mass_error,
        "exact_unbiased_flow_equalities": unbiased,
        "k3_closed_formula_verified": True,
        "LP_flow_checks": 64 * 9,
        "max_LP_error": lp_error,
        "second_moment_witness": witness,
        "nonmonotone_ratio_example": ratio_counter,
    }
    (ROOT / "results/theory_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
