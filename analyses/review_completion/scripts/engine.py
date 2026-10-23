from pathlib import Path
import ctypes, subprocess, os, sys
import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "4")
LIB = None


def calc(X, L, ks, plugin=False, P=None):
    global LIB
    if LIB is None:
        src = HERE / "fast_integrin.cpp"
        so = HERE / "fast_integrin.so"
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
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        LIB.fast5.argtypes = [
            i64,
            i64,
            f64,
            ctypes.c_int,
            i32,
            ctypes.c_int,
            i32,
            ctypes.c_int,
            f64,
        ]
        LIB.fast5.restype = None
    X = np.ascontiguousarray(X, dtype=np.int64)
    L = np.ascontiguousarray(np.broadcast_to(L, len(X)), dtype=np.int64)
    ks = np.ascontiguousarray(ks, dtype=np.int32)
    P = np.ascontiguousarray(X / L[:, None] if P is None else P, dtype=np.float64)
    assert (
        X.shape == P.shape
        and X.shape[1] == 5
        and np.all(X >= 0)
        and np.all(X.sum(1) <= L)
        and np.all(P >= 0)
        and np.all(P.sum(1) < 1)
    )
    assert plugin or np.min(L) >= np.max(ks)
    caps = np.tile(ks, (len(X), 1)).astype(np.int32)
    bound = np.zeros_like(caps, dtype=float)
    if plugin:
        rho = P.sum(1)
        for j, k in enumerate(ks):
            caps[:, j] = np.minimum(k, binom.isf(1e-15, k, rho)).astype(np.int32)
            while True:
                bad = (caps[:, j] < k) & (binom.sf(caps[:, j], k, rho) > 1e-26)
                if not bad.any():
                    break
                caps[bad, j] += 1
            # E[N 1(N>m)]/(2k) = rho/2 P(Bin(k-1,rho)>=m)
            bound[:, j] = rho * 0.5 * binom.sf(caps[:, j] - 1, k - 1, rho)
    out = np.empty((len(X), len(ks), 5))
    LIB.fast5(X, L, P, len(X), ks, len(ks), caps, int(plugin), out)
    return out, bound


def score(f):
    f = np.asarray(f)
    return np.divide(
        f[..., 1:].min(-1), f[..., 0], out=np.full(f.shape[:-1], np.nan), where=f[..., 0] > 0
    )


def population(s, ks):
    p = np.array([[j, 1, 1, 1, 1] for j in range(4)]) * s
    f, b = calc(np.zeros((4, 5), int), np.ones(4, int) * max(ks), ks, plugin=True, P=p)
    A = 0.25 * f[0] + 0.75 * f[2]
    B = 0.75 * f[1] + 0.25 * f[3]
    return np.stack([A, B]), np.stack([0.25 * b[0] + 0.75 * b[2], 0.75 * b[1] + 0.25 * b[3]])


if __name__ == "__main__":
    import time

    sys.path.insert(0, str(HERE.parents[1] / "molecular_resolution/scripts"))
    from resolution import conditional_flows

    rng = np.random.default_rng(26101873)
    X = rng.poisson(3, (200, 5))
    L = rng.integers(500, 10000, 200)
    ks = [2, 8, 32, 128, 500]
    f, b = calc(X, L, ks)
    old, e = conditional_flows(X, L, ks)
    print("max_exact_difference", np.max(abs(f - old)))
    assert np.allclose(f, old, rtol=1e-10, atol=1e-15)
    sys.path.insert(0, str(HERE.parents[1] / "strong_comparators/scripts"))
    from plugin import multinomial_flows

    f, b = calc(X, L, [8], plugin=True)
    old, e = multinomial_flows(X / L[:, None], 8)
    print("max_plugin_difference", np.max(abs(f[:, 0] - old)))
    assert np.allclose(f[:, 0], old, rtol=1e-10, atol=1e-15)
    for s in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        f, b = population(s, ks)
        print(s, score(f)[0] - score(f)[1], b.max())
    X = rng.poisson(10, (10000, 5))
    L = np.full(len(X), 10000)
    t = time.perf_counter()
    calc(X, L, ks)
    print("10000cells_exact_seconds", time.perf_counter() - t)
