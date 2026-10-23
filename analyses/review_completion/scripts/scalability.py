"""Exact integrand benchmarks, including cyclic graphs; separate from integration."""

from pathlib import Path
import time, json, math
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from general_graph import flow_values, integrate

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(26101880)
edgefile = pd.read_csv(ROOT / "inputs/integrin_edges_with_provenance.csv")
genes = np.load(ROOT / "inputs/atlas_integrins_all_cells.npz")["genes"].tolist()
E = np.array(
    [
        (genes.index(a), genes.index(b))
        for a, b in edgefile[["alpha", "beta"]].itertuples(index=False, name=None)
    ]
)
X = rng.integers(0, 12, (40, 26))
f = flow_values(X, E)
checks = 0
for i, x in enumerate(X):
    for j in range(25):
        ee = E if j == 0 else np.delete(E, j - 1, 0)
        C = np.zeros((26, len(ee)))
        for l, (a, b) in enumerate(ee):
            C[a, l] = C[b, l] = 1
        v = -linprog(-np.ones(len(ee)), A_ub=C, b_ub=x, bounds=(0, None), method="highs").fun
        assert abs(v - f[i, j]) < 1e-9
        checks += 1
rows = []


def generic_values(X, edges):
    n, d = X.shape
    e = len(edges)
    left = np.unique(edges[:, 0])
    right = np.unique(edges[:, 1])
    assert not np.intersect1d(left, right).size
    out = np.zeros((n, e + 1))
    source = d
    sink = d + 1
    for i, x in enumerate(X):
        base = np.zeros((d + 2, d + 2), np.int64)
        base[source, left] = x[left]
        base[right, sink] = x[right]
        base[edges[:, 0], edges[:, 1]] = int(x.sum()) + 1
        for j in range(e + 1):
            c = base.copy()
            if j:
                c[edges[j - 1, 0], edges[j - 1, 1]] = 0
            out[i, j] = maximum_flow(csr_matrix(c), source, sink).flow_value
    return out


for d, kind in [(26, "integrin_forest"), (20, "cyclic_bipartite"), (50, "cyclic_bipartite")]:
    if d == 26:
        edges = E
    else:
        h = d // 2
        edges = np.array(sorted({(a, h + (a + j) % h) for a in range(h) for j in [0, 1, 3]}))
    for k in [128, 500]:
        counts = rng.multinomial(k, np.r_[np.full(d, 0.001), 1 - d * 0.001], size=120)[:, :d]
        t = time.perf_counter()
        out = flow_values(counts, edges) if d == 26 else generic_values(counts, edges)
        elapsed = time.perf_counter() - t
        assert np.all(out[:, 1:] <= out[:, :1]) and np.all(out >= 0)
        rows.append(
            dict(
                graph=kind,
                genes=d,
                edges=len(edges),
                k=k,
                integrand_states=120,
                all_deletion_evaluations=120 * (len(edges) + 1),
                seconds=elapsed,
                full_enumeration_states=str(math.comb(k + d, d)),
                operation="intact and all edge-deleted flow for sampled count states; not full expectation integration",
            )
        )
        if d != 26:
            for i in range(10):
                C = np.zeros((d, len(edges)))
                for l, (a, b) in enumerate(edges):
                    C[a, l] = C[b, l] = 1
                v = -linprog(
                    -np.ones(len(edges)), A_ub=C, b_ub=counts[i], bounds=(0, None), method="highs"
                ).fun
                assert abs(v - out[i, 0]) < 1e-9
                checks += 1
pd.DataFrame(rows).to_csv(ROOT / "results/scalability_integrand_benchmark.csv", index=False)
(ROOT / "results/full_graph_solver_checks.json").write_text(
    json.dumps(
        dict(
            independent_LP_checks=checks,
            all_passed=True,
            cyclic_graph_solver="SciPy maximum_flow; no forest formula applied to cycles",
            integration_distinction="Full expectation exact enumeration is combinatorial; the scalable routine integrates low occupancy exactly and uses controlled conditional MC in the residual",
        ),
        indent=2,
    )
    + "\n"
)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
