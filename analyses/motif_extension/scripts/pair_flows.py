"""Exact bivariate conditional flows; no latent distribution is fitted.

For each cell/edge/k, return E[min(M_u,M_v)/k | X,L] with
M multivariate hypergeometric. Each edge is evaluated in isolation.
Summing these values may allocate the same vertex capacity more than once;
this is an explicit mechanistic comparator, not a replacement full-graph flow.
"""

from pathlib import Path
import ctypes
import os
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
LIB = None


def integrate_pairs(X, L, edges, ks):
    global LIB
    X = np.ascontiguousarray(X, np.int64)
    L = np.ascontiguousarray(L, np.int64)
    edges = np.ascontiguousarray(edges, np.int32)
    ks = np.ascontiguousarray(ks, np.int32)
    assert X.ndim == 2 and L.shape == (len(X),)
    assert np.all(X >= 0) and np.all(X.sum(1) <= L)
    assert np.all(L >= ks.max()) and np.all(ks >= 2)
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert np.all(edges >= 0) and np.all(edges < X.shape[1])
    assert np.all(edges[:, 0] != edges[:, 1])
    if LIB is None:
        src = HERE / "pair_flows.cpp"
        so = HERE / "libpair_flows.so"
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
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        LIB = ctypes.CDLL(str(so))
        al = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        ai = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        af = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        LIB.pair_flows.argtypes = [
            al,
            al,
            ctypes.c_int,
            ctypes.c_int,
            ai,
            ctypes.c_int,
            ai,
            ctypes.c_int,
            af,
        ]
        LIB.pair_flows.restype = None
    out = np.empty((len(X), len(ks), len(edges)))
    LIB.pair_flows(X, L, len(X), X.shape[1], edges, len(edges), ks, len(ks), out)
    return out


def multinomial_pairs(P, k):
    """Multinomial pair expectation; omitted occupancy error <= 1e-16.

    The tail rule uses the size-biased binomial first moment and a Chernoff
    bound. It bounds truncation only, in addition to floating-point error.
    """
    global LIB
    P = np.ascontiguousarray(P, np.float64)
    assert P.ndim == 2 and P.shape[1] == 2 and np.all(P >= 0) and np.all(P.sum(1) <= 1)
    if LIB is None:
        integrate_pairs(np.array([[1, 1]]), np.array([2]), np.array([[0, 1]]), [2])
    af = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    LIB.multinomial_pairs.argtypes = [af, ctypes.c_int, ctypes.c_int, af]
    LIB.multinomial_pairs.restype = None
    out = np.empty(len(P))
    LIB.multinomial_pairs(P, len(P), int(k), out)
    return out


def wedge_moments(X, L, edges):
    """Unbiased distinct-gene cubic products for each graph wedge."""
    from itertools import combinations

    X = np.asarray(X, dtype=float)
    L = np.asarray(L, dtype=float)
    adjacency = [[] for _ in range(X.shape[1])]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    wedges = [(u, v, w) for v, nb in enumerate(adjacency) for u, w in combinations(nb, 2)]
    den = L * (L - 1) * (L - 2)
    values = (
        np.stack([X[:, u] * X[:, v] * X[:, w] / den for u, v, w in wedges], axis=1)
        if wedges
        else np.zeros((len(X), 0))
    )
    return wedges, values
