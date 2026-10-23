"""Exact occupancy-2/3 strata and unbiased conditional Monte Carlo remainder.

Analytic strata apply to simple bipartite graphs. The supplied implementation
uses an O(V+E) rerooted weighted-vertex-cover algorithm for forests; cyclic
graphs use the separate max-flow benchmark. Integration is NOT exact overall.
"""

from pathlib import Path
import ctypes, subprocess, os, time, json
import numpy as np
from scipy.stats import hypergeom

HERE = Path(__file__).resolve().parent
LIB = None


def library():
    global LIB
    if LIB is None:
        src = HERE / "forest_residual.cpp"
        so = HERE / "forest_residual.so"
        if not so.exists() or src.stat().st_mtime > so.stat().st_mtime:
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-fopenmp",
                    "-shared",
                    "-fPIC",
                    str(src),
                    "-o",
                    str(so),
                ],
                check=True,
            )
        LIB = ctypes.CDLL(str(so))
        ai = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        al = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        af = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        LIB.forest_values.argtypes = [ai, ctypes.c_int, ctypes.c_int, ai, ctypes.c_int, af]
        LIB.forest_values.restype = None
        LIB.forest_mc.argtypes = [
            al,
            al,
            ctypes.c_int,
            ctypes.c_int,
            ai,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            af,
        ]
        LIB.forest_mc.restype = None
    return LIB


def validate_forest(d, edges):
    parent = list(range(d))

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra == rb:
            raise ValueError("Forest evaluator requires an acyclic graph")
        parent[ra] = rb


def flow_values(X, edges):
    X = np.ascontiguousarray(X, np.int32)
    edges = np.ascontiguousarray(edges, np.int32)
    validate_forest(X.shape[1], edges)
    out = np.empty((len(X), len(edges) + 1))
    library().forest_values(X, len(X), X.shape[1], edges, len(edges), out)
    return out


def exact_low_strata(X, L, k, edges):
    X = np.asarray(X, dtype=float)
    S = X.sum(1)
    n = len(X)
    e = len(edges)
    pair = X[:, edges[:, 0]] * X[:, edges[:, 1]]
    D2 = np.maximum(S * (S - 1), 1)
    D3 = np.maximum(S * (S - 1) * (S - 2), 1)
    h = pair / D2[:, None]
    edge3 = 2 * h - pair * (X[:, edges[:, 0]] + X[:, edges[:, 1]] - 2) / D3[:, None]
    wedges = np.zeros((n, e))
    totalw = np.zeros(n)
    for j in range(e):
        for l in range(j + 1, e):
            u = set(edges[j]) | set(edges[l])
            if len(u) == 3:
                w = np.prod(X[:, list(u)], axis=1) / D3
                totalw += w
                wedges[:, j] += w
                wedges[:, l] += w
    phi2 = np.column_stack([h.sum(1), h.sum(1)[:, None] - h])
    f3 = edge3.sum(1) - 2 * totalw
    phi3 = np.column_stack([f3, f3[:, None] - edge3 + 2 * wedges])
    phi3[S < 3] = 0
    p2 = hypergeom.pmf(2, L, S, k)
    p3 = hypergeom.pmf(3, L, S, k)
    low = 2 / k * p2[:, None] * phi2 + 3 / k * p3[:, None] * phi3
    tailbound = (
        S / (2 * L) * hypergeom.sf(2, L - 1, np.maximum(S - 1, 0), k - 1) if k >= 4 else np.zeros(n)
    )
    return low, tailbound


def integrate(X, L, k, edges, batches=8, draws=8, seed=26101875):
    X = np.ascontiguousarray(X, np.int64)
    L = np.ascontiguousarray(L, np.int64)
    edges = np.ascontiguousarray(edges, np.int32)
    assert np.all(X >= 0) and np.all(X.sum(1) <= L) and np.all(L >= k)
    validate_forest(X.shape[1], edges)
    low, bound = exact_low_strata(X, L, k, edges)
    out = np.zeros((len(X), batches, len(edges) + 1))
    if k >= 4:
        library().forest_mc(
            X, L, len(X), X.shape[1], edges, len(edges), k, batches, draws, seed, out
        )
    out += low[:, None, :]
    return out, bound


if __name__ == "__main__":
    from scipy.optimize import linprog
    import sys

    sys.path.insert(0, str(HERE.parents[1] / "molecular_resolution/scripts"))
    from resolution import conditional_flows

    rng = np.random.default_rng(26101875)
    E = np.array([(0, 2), (1, 2), (1, 3), (1, 4)])
    X = rng.integers(0, 8, (120, 5))
    L = rng.integers(500, 10000, 120)
    f = flow_values(X, E)
    checks = 0
    for i, x in enumerate(X):
        for j in range(5):
            edges = E if j == 0 else np.delete(E, j - 1, axis=0)
            C = np.zeros((5, len(edges)))
            for l, (a, b) in enumerate(edges):
                C[a, l] = C[b, l] = 1
            truth = -linprog(
                -np.ones(len(edges)), A_ub=C, b_ub=x, bounds=(0, None), method="highs"
            ).fun
            assert abs(f[i, j] - truth) < 1e-10
            checks += 1
    for k in [2, 3]:
        g, _ = integrate(X, L, k, E)
        old, _ = conditional_flows(X, L, [k])
        assert np.allclose(g.mean(1), old[:, 0], rtol=1e-9, atol=1e-14)
        checks += 120
    old, _ = conditional_flows(X, L, [500])
    g, _ = integrate(X, L, 500, E, batches=32, draws=128)
    error = g.mean((0, 1)) - old[:, 0].mean(0)
    se = g.mean(0).std(0, ddof=1) / np.sqrt(32)
    assert np.all(abs(error) < 6 * se + 1e-13)
    report = dict(
        independent_checks=checks,
        max_flow_lp_passed=True,
        low_strata_exact_passed=True,
        aggregate_mc_error=error.tolist(),
        aggregate_mc_se=se.tolist(),
        all_passed=True,
    )
    (HERE.parent / "results/general_graph_checks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report, flush=True)
