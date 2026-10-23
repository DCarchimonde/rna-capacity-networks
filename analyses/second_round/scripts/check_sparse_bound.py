"""Independent finite-state checks of the sparse-occupancy lower sandwich."""

from pathlib import Path
from itertools import product, combinations
from math import factorial
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "analyses/motif_extension/scripts"))
from pair_flows import multinomial_pairs


def compositions(total, d):
    if d == 1:
        yield (total,)
    else:
        for x in range(total + 1):
            for tail in compositions(total - x, d - 1):
                yield (x,) + tail


def evaluate(edges, d, k, rho):
    p = np.full(d, rho / d)
    prob = np.r_[p, 1 - rho]
    covers = np.array(
        [bits for bits in product([0, 1], repeat=d) if all(bits[u] or bits[v] for u, v in edges)]
    )
    wedges = [
        (u, v, w)
        for v in range(d)
        for u, w in combinations([b if a == v else a for a, b in edges if a == v or b == v], 2)
    ]
    w = sum(p[u] * p[v] * p[t] for u, v, t in wedges)
    D = B = 0.0
    for m in compositions(k, d + 1):
        probability = factorial(k)
        for x, q in zip(m, prob):
            probability *= q**x / factorial(x)
        x = np.array(m[:d])
        full = np.min(covers @ x)
        independent = sum(min(x[u], x[v]) for u, v in edges)
        D += probability * (independent - full) / k
        B += probability * independent / k
    upper = (k - 1) * (k - 2) * w
    lower = upper * (1 - rho) ** (k - 3)
    tol = 2e-14 * max(upper, B, 1e-20)
    assert D >= lower - tol and D <= min(B, upper) + tol
    if k == 3:
        assert abs(D - upper) <= tol
    return dict(k=k, rho=rho, D=D, B=B, lower=lower, upper=upper)


graphs = dict(
    wedge=([(0, 1), (0, 2)], 3),
    matching=([(0, 2), (1, 3)], 4),
    path=([(0, 2), (1, 2), (1, 3)], 4),
    K23=([(u, v) for u in [0, 1] for v in [2, 3, 4]], 5),
)
checks = [
    dict(graph=name, **evaluate(edges, d, k, rho))
    for name, (edges, d) in graphs.items()
    for k in [3, 4, 8]
    for rho in [1e-4, 0.02, 0.4, 1.0]
]
b = float(6 * multinomial_pairs(np.array([[0.2, 0.2]]), 500)[0])
assert b > 1
report = dict(
    checks=len(checks),
    all_passed=True,
    examples=checks,
    counterexample_to_B_le_rho=dict(graph="K2,3", probabilities=[0.2] * 5, rho=1, k=500, B=b),
    proof="exact three-panel-molecule wedge event gives the lower bound; other occupancies have nonnegative loss",
)
(ROOT / "results/bound_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
