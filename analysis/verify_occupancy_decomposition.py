"""Exact check of the new S11 occupancy identity; only the Python standard library."""

from pathlib import Path
from fractions import Fraction as Q
from math import factorial, comb
import json


def states(total, d):
    if d == 1:
        yield (total,)
    else:
        for x in range(total + 1):
            for rest in states(total - x, d - 1):
                yield (x,) + rest


def flow(m):
    a, v, b, c, d = m[:5]
    return [
        min(a + v, v + b, b + c + d),
        min(v, b + c + d),
        min(a, b) + min(v, c + d),
        min(a + v, v + b, b + d),
        min(a + v, v + b, b + c),
    ]


def expectation(k, p):
    ans = [Q(0)] * 5
    for m in states(k, len(p)):
        prob = Q(factorial(k))
        for mi, pi in zip(m, p):
            prob *= pi**mi / factorial(mi)
        for j, f in enumerate(flow(m)):
            ans[j] += prob * f / k
    return ans


q = tuple(Q(x, 9) for x in [1, 2, 3, 1, 2])
checks = []
psi = {n: expectation(n, q) for n in range(1, 9)}
for rho in [Q(1, 100), Q(1, 10), Q(3, 5)]:
    for k in [2, 3, 4, 8]:
        direct = expectation(k, tuple(rho * x for x in q) + (1 - rho,))
        decomposed = [
            sum(
                Q(comb(k, n)) * rho**n * (1 - rho) ** (k - n) * Q(n, k) * psi[n][j]
                for n in range(1, k + 1)
            )
            for j in range(5)
        ]
        for j in range(5):
            assert direct[j] == decomposed[j]
            checks.append(dict(rho=str(rho), k=k, component=j, equal=True))
r = Path(__file__).resolve().parents[1]
(r / "review/occupancy_identity_verification.json").write_text(
    json.dumps(
        {
            "checks": len(checks),
            "all_passed": True,
            "arithmetic": "exact rational",
            "details": checks,
        },
        indent=2,
    )
    + "\n"
)
print(len(checks), "exact occupancy-identity checks passed")
