"""Independent checks of the displayed graph and sampling identities.

Uses exhaustive vertex covers, symbolic polynomials, integer multinomial sums,
and direct molecule subsets. Does not import the analysis implementation.
"""

from pathlib import Path
from itertools import product, combinations
from fractions import Fraction as Q
from math import factorial, comb
import json, datetime
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
edges = ((0, 2), (1, 2), (1, 3), (1, 4))
all_edges = tuple(product((0, 1), (2, 3, 4)))
checks = {}


def check(name, ok):
    checks[name] = bool(ok)
    if not ok:
        raise AssertionError(name)


def compositions(n, r):
    if r == 1:
        yield (n,)
        return
    for j in range(n + 1):
        for rest in compositions(n - j, r - 1):
            yield (j,) + rest


def covers(G):
    return [v for v in range(32) if all((v >> g) & 1 or (v >> h) & 1 for g, h in G)]


def flow(counts, C):
    return min(sum(counts[j] for j in range(5) if (c >> j) & 1) for c in C)


Cs = [covers(edges)] + [covers(tuple(e for e in edges if e != deleted)) for deleted in edges]


def cut_formulas(x):
    a, v, b, c, d = x[:5]
    return [
        min(a + v, v + b, b + c + d),
        min(v, b + c + d),
        min(a, b) + min(v, c + d),
        min(a + v, v + b, b + d),
        min(a + v, v + b, b + c),
    ]


for x in product(range(4), repeat=5):
    check("cut_" + "".join(map(str, x)), cut_formulas(x) == [flow(x, C) for C in Cs])
p = sp.symbols("a v b c d")
r = 1 - sum(p)
ps = (*p, r)


def symbolic_expectation(G, k):
    C = covers(G)
    terms = []
    for m in compositions(k, 6):
        f = flow(m, C)
        if not f:
            continue
        coeff = sp.Rational(f * factorial(k), k)
        for count in m:
            coeff /= factorial(count)
        terms.append(coeff * sp.prod(prob**count for prob, count in zip(ps, m)))
    return sp.expand(sum(terms))


for mask in range(64):
    G = tuple(e for i, e in enumerate(all_edges) if mask >> i & 1)
    pair = sum(p[g] * p[h] for g, h in G)
    cubic = sum(p[g] * p[h] * (2 - p[g] - p[h]) for g, h in G) - 2 * sum(
        sp.prod(p[j] for j in set(e) | set(f))
        for e, f in combinations(G, 2)
        if len(set(e) & set(f)) == 1
    )
    check(f"pair_polynomial_graph_{mask}", sp.expand(symbolic_expectation(G, 2) - pair) == 0)
    check(f"cubic_polynomial_graph_{mask}", sp.expand(symbolic_expectation(G, 3) - cubic) == 0)
print("Cut and symbolic identities checked", flush=True)

# Exact population values: all probabilities have denominator 25.
witness = []
for k in (2, 3, 4, 8, 16):
    sums = {}
    for J in (0, 1, 2, 3):
        nums = (3 * J, 3, 3, 3, 3, 13 - 3 * J)
        total = [0] * 5
        for m in compositions(k, 6):
            probnum = factorial(k)
            for count in m:
                probnum //= factorial(count)
            # Divide factorials together to avoid sequential integer truncation.
            den = 1
            for count in m:
                den *= factorial(count)
            probnum = factorial(k) // den
            for q, c in zip(nums, m):
                probnum *= q**c
            if not probnum:
                continue
            for i, f in enumerate(cut_formulas(m)):
                total[i] += probnum * f
        sums[J] = total
    A = [sums[0][i] + 3 * sums[2][i] for i in range(5)]
    B = [3 * sums[1][i] + sums[3][i] for i in range(5)]
    ra = Q(min(A[1:]), A[0])
    rb = Q(min(B[1:]), B[0])
    witness.append(
        {"k": k, "R_A_exact": str(ra), "R_B_exact": str(rb), "R_A": float(ra), "R_B": float(rb)}
    )
    expected = {2: (Q(2, 3), Q(2, 3)), 3: (Q(76, 111), Q(76, 111)), 4: (Q(191, 282), Q(764, 1131))}
    if k in expected:
        check(f"witness_exact_{k}", (ra, rb) == expected[k])
    else:
        rounded = {8: (0.650917493, 0.642506598), 16: (0.620503269, 0.603958790)}
        check(
            f"witness_rounded_{k}",
            all(abs(a - b) < 5e-10 for a, b in zip((float(ra), float(rb)), rounded[k])),
        )
check(
    "first_two_J_moments",
    Q(3, 4) * 2 == Q(3, 4) * 1 + Q(1, 4) * 3 and Q(3, 4) * 4 == Q(3, 4) * 1 + Q(1, 4) * 9,
)
check(
    "latent_limits",
    Q(1, 1) / (Q(1, 4) * 1 + Q(3, 4) * 2) == Q(4, 7)
    and Q(1, 1) / (Q(3, 4) * 2 + Q(1, 4) * 2) == Q(1, 2),
)

# All subsets, conditional hypergeometric expression and factorial identity.
for X in [(2, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1), (0, 2, 0, 1, 1, 2)]:
    labels = [h for h, count in enumerate(X) for _ in range(count)]
    L = len(labels)
    prior = [Q(0)] * 5
    for k in range(2, L + 1):
        direct = [Q(0)] * 5
        hyper = [Q(0)] * 5
        for subset in combinations(range(L), k):
            m = [0] * 6
            for i in subset:
                m[labels[i]] += 1
            for j, f in enumerate(cut_formulas(m)):
                direct[j] += Q(f, k * comb(L, k))
        norm = Q(0)
        for m in compositions(k, 6):
            if any(a > b for a, b in zip(m, X)):
                continue
            w = Q(1, comb(L, k))
            for x, a in zip(X, m):
                w *= comb(x, a)
            norm += w
            for j, f in enumerate(cut_formulas(m)):
                hyper[j] += w * Q(f, k)
        check(f"subset_equivalence_{X}_{k}", direct == hyper and norm == 1)
        check(f"conditional_monotonicity_{X}_{k}", all(a <= b for a, b in zip(prior, hyper)))
        if k == 2:
            check(
                f"factorial_pair_{X}",
                hyper[0] == sum(Q(X[g] * X[h], L * (L - 1)) for g, h in edges),
            )
        prior = hyper

# Unbiasedness by exact count-law sums, including unequal L.
P = (Q(1, 5), Q(1, 5), Q(1, 5), Q(1, 10), Q(1, 10), Q(1, 5))
for L in (2, 3, 4):
    for k in range(2, min(L, 3) + 1):
        unconditional = [Q(0)] * 5
        target = [Q(0)] * 5
        for m in compositions(k, 6):
            w = Q(factorial(k))
            for q, c in zip(P, m):
                w *= q**c / factorial(c)
            for j, f in enumerate(cut_formulas(m)):
                target[j] += w * Q(f, k)
        for X in compositions(L, 6):
            prob = Q(factorial(L))
            for q, c in zip(P, X):
                prob *= q**c / factorial(c)
            for m in compositions(k, 6):
                if any(a > b for a, b in zip(m, X)):
                    continue
                w = prob / comb(L, k)
                for x, a in zip(X, m):
                    w *= comb(x, a)
                for j, f in enumerate(cut_formulas(m)):
                    unconditional[j] += w * Q(f, k)
        check(f"exact_unbiasedness_L{L}_k{k}", unconditional == target)

# Cross-cell U is the ordered distinct-cell product average.
ys = [(Q(1, 5), Q(1, 7)), (Q(2, 5), Q(3, 7)), (Q(1, 6), Q(1, 3))]
n = len(ys)
u = (sum(x[0] for x in ys) * sum(x[1] for x in ys) - sum(x * y for x, y in ys)) / (n * (n - 1))
check(
    "U_distinct_cell_identity",
    u == sum(ys[i][0] * ys[j][1] for i in range(n) for j in range(n) if i != j) / (n * (n - 1)),
)
out = {
    "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "all_passed": all(checks.values()),
    "checks": len(checks),
    "cut_grid_points": 1024,
    "bipartite_graphs_checked_symbolically": 64,
    "pair_and_cubic_symbolic_identities": 128,
    "witness": witness,
    "details": checks,
    "scope": "Exact graph and sampling identities; model assumptions are stated in the manuscript. This does not validate the observation model against biological truth.",
}
(ROOT / "review/equation_checks.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=2), flush=True)
