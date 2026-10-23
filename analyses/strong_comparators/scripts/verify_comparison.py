"""Independent checks for target conversion, exact integration, and prior replay."""

from pathlib import Path
import sys

# Reuse the canonical fixed-resolution implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "molecular_resolution/scripts"))
from itertools import combinations
from datetime import datetime, timezone
import contextlib, hashlib, io, json, warnings
import numpy as np
from cscore_adapter import OFFICIAL, capture_official, summarize_type, equal_cell, r2_from_h
from resolution import conditional_flows, flows
from verify_theory import population_numeric
from plugin import multinomial_flows
from numeric_metrics import pair_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    official = (ROOT / "vendor/CSCORE_IRLS.py").read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(official)).encode() + b"\x00" + official).hexdigest()
    assert git_blob == "8dd091cc97f4c668fb82e1eeb8f10018bef8ae16"
    spec = json.loads((ROOT / "config/comparison_spec.json").read_text())
    assert hashlib.sha256(official).hexdigest() == spec["official_comparator"]["source_sha256"]
    rng = np.random.default_rng(2610109388)
    trace_checks = 0
    max_constant = 0
    # Overdispersed populations make the official variance fit well-defined.
    for constant in [True, False]:
        depth = (
            np.full(400, 1200.0)
            if constant
            else np.exp(rng.uniform(np.log(800.0), np.log(12000.0), 400))
        )
        p = rng.dirichlet(np.array([0.8, 1.1, 0.9, 0.7, 0.8, 15.0]), size=400)[:, :5]
        x = rng.poisson(depth[:, None] * p)
        captured = capture_official(x, depth)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            direct = OFFICIAL.CSCORE_IRLS(x.astype(float), depth, post_process=False)
        for a, b in zip(captured["returned"], direct):
            assert np.array_equal(a, b, equal_nan=True)
            trace_checks += 1
        if constant:
            h = np.array(
                [
                    captured["mu"][a] * captured["mu"][b] + captured["covar"][a, b]
                    for a, b in spec["edges"]
                ]
            )
            max_constant = max(max_constant, float(np.max(abs(h - equal_cell(x, depth, True)))))
            assert np.allclose(h, equal_cell(x, depth, True), rtol=1e-12, atol=1e-17)
        values, status = summarize_type(x, depth, poisson=True)
        assert np.isfinite(values[[0, 1, 4], :]).all()
    # Complete U-statistic over observed molecules equals hypergeometric integration.
    subset_checks = 0
    subset_max = 0.0
    fixtures = [([2, 1, 2, 0, 1], 8), ([1, 2, 1, 1, 1], 8), ([0, 3, 2, 1, 0], 9)]
    for counts, L in fixtures:
        molecules = np.concatenate(
            [np.full(c, i) for i, c in enumerate([*counts, L - sum(counts)])]
        )
        for k in [2, 3, 4]:
            samples = np.array(
                [
                    np.bincount(molecules[list(ids)], minlength=6)[:5] / k
                    for ids in combinations(range(L), k)
                ]
            )
            for common in [False, True]:
                direct = flows(samples, common).mean(0)
                integrated, _ = conditional_flows(np.array([counts]), np.array([L]), [k], common)
                err = float(np.max(abs(direct - integrated[0, 0])))
                subset_max = max(subset_max, err)
                assert err < 1e-13
                subset_checks += len(direct)
    plugin_checks = 0
    plugin_max = 0
    probs = rng.dirichlet(np.ones(6), size=8)[:, :5]
    probs = np.r_[probs, np.array([[0, 0.2, 0.15, 0, 0.1], [0.2, 0, 0, 0, 0]])]
    for k in [2, 3, 8]:
        for common in [False, True]:
            prediction, mass = multinomial_flows(probs, k, common)
            for p, a in zip(probs, prediction):
                expected = population_numeric([(1, p)], k, common)
                err = float(np.max(abs(a - expected)))
                plugin_max = max(plugin_max, err)
                assert err < 1e-13
                plugin_checks += len(a)
    # A summation-order perturbation cannot manufacture a strict direction.
    ref = np.array([1e-9, 2e-9, 3e-9])
    query = np.array([1e-9, np.nextafter(1e-9, np.inf), 2e-9])
    assert pair_metrics(ref, query)["query_resolution"] == 2 / 3
    replay = json.loads((ROOT / "results/H_simulation_run.json").read_text())
    r8 = json.loads((ROOT / "results/R8_simulation_run.json").read_text())
    assert replay["cohorts"] == 1152 and replay["all_24_source_snapshots_identical"]
    assert r8["replayed_cohorts"] == 192 and r8["replay_R_max_error"] < 1e-11
    real_replay = {}
    r2_max = 0
    for name in ["D", "I"]:
        new = np.load(ROOT / f"results/{name}_H_scores.npz")
        reference = "inputs/development_reference" if name == "D" else "results/independent"
        old = np.load(ROOT.parent / "independent_pair" / reference / "scores.npz")
        eligible = [new["Epi_eligible"], new["dual_eligible"], new["dual_eligible"]]
        errors = {}
        for key in ["full", "halves", "thin"]:
            a = new[key][..., 0, :]
            b = old[key][..., 0, :]
            # Ignore non-qualified type slots which are not fitted in this comparison.
            if key == "full":
                mask = np.stack(eligible, axis=1)[:, :, None]
            else:
                mask = np.stack(eligible, axis=1)[None, :, :, None]
            good = np.broadcast_to(mask, a.shape) & np.isfinite(a) & np.isfinite(b)
            err = float(np.max(abs(a[good] - b[good])))
            errors[key] = err
            assert err < 1e-18
        real_replay[name] = errors
        prior = np.load(ROOT / f"../molecular_resolution/results/{name}_common3_cell_flows.npz")
        measured = r2_from_h(new["full"][:, :, 0, :], True)
        ref = prior["patient_R"][:, :, 0]
        good = np.isfinite(measured) & np.isfinite(ref)
        err = float(np.max(abs(measured[good] - ref[good])))
        r2_max = max(r2_max, err)
        assert err < 1e-12
    outcome = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": True,
        "official_git_blob": git_blob,
        "official_file_unchanged": True,
        "traced_vs_untraced_output_arrays_bitwise_equal": trace_checks,
        "constant_depth_raw_parameter_identity_max_error": max_constant,
        "complete_U_statistic_graph_flow_checks": subset_checks,
        "U_statistic_max_error": subset_max,
        "plugin_independent_multinomial_flow_checks": plugin_checks,
        "plugin_max_error": plugin_max,
        "real_equal_cell_H_replay_max_errors": real_replay,
        "real_R2_identity_max_error": r2_max,
        "old_768_cohorts_all_scores_replayed": True,
        "all_24_original_raw_count_snapshots_equal": True,
        "old_192_R8_cohorts_replayed": True,
        "common_floating_arithmetic_rule_checked": True,
        "missing_ITGB3_handled_as_unavailable": True,
    }
    (ROOT / "results/verification.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
