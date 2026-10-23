"""Unchanged official CS-CORE core; expose its raw moment parameters by tracing.

The practical fallback is reported separately from the official raw estimator.
"""

from pathlib import Path
import contextlib, importlib.util, io, sys, warnings
import numpy as np

HERE = Path(__file__).resolve().parents[1]
EDGES = [(0, 2), (1, 2), (1, 3), (1, 4)]
METHODS = [
    "equal_cell_moment",
    "samecell_fraction",
    "CSCORE_graph_raw",
    "CSCORE_graph_projected",
    "CSCORE_graph_fallback",
    "CSCORE_pair_fallback",
]


class NumpyCompat:
    Inf = np.inf

    def __getattr__(self, key):
        return getattr(np, key)


spec = importlib.util.spec_from_file_location(
    "official_CSCORE_IRLS", HERE / "vendor/CSCORE_IRLS.py"
)
OFFICIAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OFFICIAL)
OFFICIAL.np = NumpyCompat()


def capture_official(x, depth):
    """Use exactly the official fit and save local parameters on its return."""
    saved = {}
    fn = OFFICIAL.CSCORE_IRLS

    def local_trace(frame, event, arg):
        if event == "return":
            for key in ["mu", "covar", "neg_inds", "j", "delta"]:
                if key in frame.f_locals:
                    value = frame.f_locals[key]
                    saved[key] = value.copy() if isinstance(value, np.ndarray) else value
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and frame.f_code is fn.__code__:
            return local_trace
        return None

    previous = sys.gettrace()
    try:
        sys.settrace(global_trace)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore", RuntimeWarning)
            result = fn(
                np.asarray(x, dtype=float), np.asarray(depth, dtype=float), post_process=False
            )
    finally:
        sys.settrace(previous)
    saved["returned"] = result
    return saved


def equal_cell(x, depth, poisson=False):
    x = np.asarray(x, float)
    l = np.asarray(depth, float)
    den = l * l if poisson else l * (l - 1)
    if np.ptp(l) == 0:
        return np.array([np.mean(x[:, a] * x[:, b]) / den[0] for a, b in EDGES])
    return np.array([(x[:, a] * x[:, b] / den).mean() for a, b in EDGES])


def fit_panel(x, depth, panel):
    x = np.asarray(x, float)
    depth = np.asarray(depth, float)
    active = [i for i in panel if x[:, i].sum() > 0]
    h = np.full(4, np.nan)
    # A measured all-zero endpoint has boundary point estimate H=0.
    for e, (a, b) in enumerate(EDGES):
        if a in panel and b in panel and (a not in active or b not in active):
            h[e] = 0.0
    status = {
        "panel": ",".join(map(str, panel)),
        "active_genes": len(active),
        "zero_count_genes": len(panel) - len(active),
        "fit_status": "not_needed_zero_edges",
        "j": 0,
        "delta": 0.0,
        "negative_variance_genes": 0,
        "negative_H_edges": 0,
    }
    if len(active) < 2:
        return h, status
    try:
        result = capture_official(x[:, active], depth)
        mu, cov = result["mu"], result["covar"]
        pos = {g: i for i, g in enumerate(active)}
        for e, (a, b) in enumerate(EDGES):
            if a in pos and b in pos:
                h[e] = mu[pos[a]] * mu[pos[b]] + cov[pos[a], pos[b]]
        raw_h = h.copy()
        # At constant depth, the official moment regressions have constant
        # cell weights. The raw parameter sum is exactly the observed product
        # mean divided by L^2. Evaluate that identity without cancellation.
        if np.ptp(depth) == 0:
            for e, (a, b) in enumerate(EDGES):
                if a in pos and b in pos and np.isfinite(h[e]):
                    h[e] = np.mean(x[:, a] * x[:, b]) / (depth[0] ** 2)
            finite = np.isfinite(raw_h) & np.isfinite(h)
            status["constant_depth_identity_used"] = True
            status["identity_max_abs_discrepancy"] = (
                float(np.max(abs(raw_h[finite] - h[finite]))) if finite.any() else 0.0
            )
            status["raw_parameter_sum_negative_edges"] = int((raw_h[np.isfinite(raw_h)] < 0).sum())
        status.update(
            fit_status="returned",
            j=int(result["j"]),
            delta=float(result["delta"]),
            negative_variance_genes=int(result["neg_inds"].sum()),
        )
    except Exception as error:
        status.update(fit_status="exception", error_type=type(error).__name__, error=str(error))
    status["nonfinite_H_edges"] = int(
        sum(not np.isfinite(h[e]) for e, (a, b) in enumerate(EDGES) if a in panel and b in panel)
    )
    status["negative_H_edges"] = int(np.sum(h[np.isfinite(h)] < 0))
    status["actual_converged"] = (
        status["fit_status"] == "returned"
        and np.isfinite(status["delta"])
        and status["delta"] <= 0.05
    )
    return h, status


def summarize_type(x, depth, present=None, poisson=False):
    if not len(depth):
        return np.full((len(METHODS), 4), np.nan), []
    present = np.ones(5, bool) if present is None else np.asarray(present, bool)
    panel = np.flatnonzero(present).tolist()
    moment = equal_cell(x, depth, poisson)
    fraction = equal_cell(x, depth, True)
    graph, gs = fit_panel(x, depth, panel)
    pair, ps = fit_panel(x, depth, [1, 4])
    projected = np.maximum(graph, 0)
    practical = np.where(np.isfinite(graph), projected, moment)
    pair_practical = np.full(4, np.nan)
    pair_practical[3] = max(pair[3], 0) if np.isfinite(pair[3]) else moment[3]
    values = np.stack([moment, fraction, graph, projected, practical, pair_practical])
    edge_present = np.array([present[a] and present[b] for a, b in EDGES])
    values[:, ~edge_present] = np.nan
    gs["variant"] = "graph"
    ps["variant"] = "pair"
    gs["fallback_edges"] = int(np.sum(edge_present & ~np.isfinite(graph)))
    ps["fallback_edges"] = int(not np.isfinite(pair[3]))
    return values, [gs, ps]


def r2_from_h(h, common=False):
    h = np.asarray(h)[..., [0, 1, 3] if common else [0, 1, 2, 3]]
    total = h.sum(-1)
    valid = np.isfinite(h).all(-1) & (h >= 0).all(-1) & (total > 0)
    return np.divide(total - h.max(-1), total, out=np.full(total.shape, np.nan), where=valid)
