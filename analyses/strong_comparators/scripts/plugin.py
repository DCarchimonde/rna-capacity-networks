from pathlib import Path
import ctypes, subprocess, os
import numpy as np

ROOT = Path(__file__).resolve().parent
LIB = None


def multinomial_flows(p, k=8, common=False):
    global LIB
    if k < 1 or k > 16:
        raise ValueError("Only resolutions 1..16 supported by this benchmark")
    p = np.ascontiguousarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or np.any(p < 0) or np.any(p.sum(1) > 1 + 1e-12):
        raise ValueError("Invalid RNA probabilities")
    if LIB is None:
        target = ROOT / "multinomial_plugin.so"
        source = ROOT / "multinomial_plugin.cpp"
        if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-fopenmp",
                    str(source),
                    "-o",
                    str(target),
                ],
                check=True,
            )
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        LIB = ctypes.CDLL(str(target))
        af = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        LIB.multinomial_plugin.argtypes = [af, ctypes.c_int, ctypes.c_int, ctypes.c_int, af, af]
        LIB.multinomial_plugin.restype = None
    out = np.empty((len(p), 4 if common else 5))
    mass = np.empty(len(p))
    LIB.multinomial_plugin(p, len(p), k, int(common), out, mass)
    if mass.max(initial=0) > 1e-9:
        raise ArithmeticError("Multinomial probability mass failed")
    return out, mass
