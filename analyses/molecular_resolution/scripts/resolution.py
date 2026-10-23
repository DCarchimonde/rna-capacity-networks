"""RNA max-flow at fixed molecule count: exact conditional expectation."""

from pathlib import Path
import ctypes
import subprocess
import os
import numpy as np

HERE = Path(__file__).resolve().parent
EDGES = [(0, 2), (1, 2), (1, 3), (1, 4)]


def flows(x, common=False):
    a, v, b, c, d = np.moveaxis(np.asarray(x), -1, 0)
    base = np.minimum.reduce([a + v, v + b, b + d if common else b + c + d])
    fab = np.minimum(v, b + d if common else b + c + d)
    fvb = np.minimum(a, b) + np.minimum(v, d if common else c + d)
    if common:
        return np.stack([base, fab, fvb, np.minimum.reduce([a + v, v + b, b])], axis=-1)
    return np.stack(
        [
            base,
            fab,
            fvb,
            np.minimum.reduce([a + v, v + b, b + d]),
            np.minimum.reduce([a + v, v + b, b + c]),
        ],
        axis=-1,
    )


def compile_library():
    target = HERE / "flow_resolution.so"
    source = HERE / "flow_resolution.cpp"
    if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-fopenmp",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(target),
            ],
            check=True,
        )
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    lib = ctypes.CDLL(str(target))
    a64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    a32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    af = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.flow_resolution.argtypes = [a64, a64, ctypes.c_int, a32, ctypes.c_int, ctypes.c_int, af, af]
    lib.flow_resolution.restype = None
    return lib


LIB = None


def conditional_flows(x, depth, ks, common=False):
    global LIB
    if LIB is None:
        LIB = compile_library()
    x = np.ascontiguousarray(x, dtype=np.int64)
    depth = np.ascontiguousarray(depth, dtype=np.int64)
    ks = np.ascontiguousarray(ks, dtype=np.int32)
    if x.shape != (len(depth), 5) or np.any(x < 0) or np.any(x.sum(1) > depth):
        raise ValueError("Invalid count categories")
    if np.any(ks < 1) or np.any(depth < ks.max()):
        raise ValueError("Every observed depth must be >= every k")
    out = np.empty((len(x), len(ks), 4 if common else 5))
    errors = np.empty((len(x), len(ks)))
    LIB.flow_resolution(x, depth, len(x), ks, len(ks), int(common), out, errors)
    if errors.max(initial=0) > 1e-8:
        raise ArithmeticError(f"Hypergeometric normalization error {errors.max()}")
    return out, errors


def r_from_flows(a):
    """Ratio of average flows, not average per-cell ratios."""
    a = np.asarray(a)
    return np.divide(
        a[..., 1:].min(axis=-1), a[..., 0], out=np.full(a.shape[:-1], np.nan), where=a[..., 0] > 0
    )


def h_flows(x, depth, common=False):
    h = np.stack([x[:, g] * x[:, h] / (depth * (depth - 1)) for g, h in EDGES], axis=-1)
    if common:
        h = h[:, [0, 1, 3]]
    s = h.sum(axis=-1)
    return np.column_stack([s, s[:, None] - h])


def random_rarefaction(x, depth, k, rng, draws=1, common=False):
    acc = np.zeros((len(depth), 4 if common else 5))
    for _ in range(draws):
        remaining = depth.copy()
        needed = np.full(len(depth), k, dtype=np.int64)
        counts = np.zeros_like(x)
        for j in range(5):
            counts[:, j] = rng.hypergeometric(x[:, j], remaining - x[:, j], needed)
            remaining -= x[:, j]
            needed -= counts[:, j]
        acc += flows(counts / k, common)
    return acc / draws
