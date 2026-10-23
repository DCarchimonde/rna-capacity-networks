"""Run the manuscript analyses from the supplied count projections."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def command(path, *arguments):
    return [str(path), *map(str, arguments)]


CHECKS = [
    command("scripts/verify_equations_independent.py"),
    command("analyses/molecular_resolution/scripts/verify_theory.py"),
    command("analyses/strong_comparators/scripts/verify_comparison.py"),
    command("analyses/review_completion/scripts/engine.py"),
    command("analyses/review_completion/scripts/general_graph.py"),
    command("analyses/review_completion/scripts/scalability.py"),
    command("analyses/motif_extension/scripts/verify_theory.py"),
    command("analyses/second_round/scripts/check_sparse_bound.py"),
    command("analysis/verify_occupancy_decomposition.py"),
    command("scripts/check_results.py"),
]
FIGURES = [
    command("scripts/make_figures.py"),
    command("analyses/review_completion/scripts/make_figures.py"),
    command("integration/make_current_figures.py"),
    command("analyses/motif_extension/scripts/make_figures.py"),
]
SUMMARIES = [
    command("integration/propagate_refinement.py"),
    command("analyses/motif_extension/scripts/analyze_competition.py"),
    command("analyses/review_completion/scripts/phenotype.py"),
    command("analyses/review_completion/scripts/pairout_auc.py"),
    command("analyses/review_completion/scripts/revision_summary.py"),
    command("analyses/second_round/scripts/deletion_edges.py"),
    command("analysis/review_diagnostics.py"),
    command("scripts/sync_tables.py"),
    command("scripts/check_results.py"),
]
CORE = [
    command("analyses/review_completion/scripts/real_expansion.py"),
    command("analyses/motif_extension/scripts/refine_close_orders.py"),
    command("analyses/second_round/scripts/paired_power.py"),
    *SUMMARIES,
    *FIGURES,
]
BASELINE = [
    command(
        "analyses/independent_pair/scripts/analyze_h.py",
        "--cache",
        "data/development_counts.npz",
        "--spec",
        "analyses/independent_pair/config/execution_spec.json",
        "--out-dir",
        "analyses/independent_pair/inputs/development_reference",
        "--input-kind",
        "development_replay",
    ),
    command(
        "analyses/independent_pair/scripts/analyze_h.py",
        "--cache",
        "data/independent_counts.npz",
        "--spec",
        "analyses/independent_pair/config/execution_spec.json",
        "--out-dir",
        "analyses/independent_pair/results/independent",
        "--input-kind",
        "independent",
    ),
    command(
        "analyses/independent_pair/scripts/summarize_replication.py",
        "--root",
        "analyses/independent_pair",
        "--development-results",
        "analyses/independent_pair/inputs/development_reference",
    ),
    command(
        "analyses/independent_pair/scripts/decompose_epithelial_subtypes.py",
        "--root",
        "analyses/independent_pair",
    ),
    command("analyses/molecular_resolution/scripts/verify_theory.py"),
    command("analyses/molecular_resolution/scripts/run_simulations.py"),
    command("analyses/molecular_resolution/scripts/analyze_cohorts.py"),
    command("analyses/strong_comparators/scripts/benchmark_H.py"),
    command("analyses/strong_comparators/scripts/benchmark_R8.py"),
    command("analyses/strong_comparators/scripts/analyze_comparison.py"),
    command("scripts/sync_tables.py"),
]
SIMULATIONS = [
    command("analyses/review_completion/scripts/sparse_experiments.py"),
    command("analyses/review_completion/scripts/detection_power.py"),
    command("analyses/review_completion/scripts/near_tie_background.py"),
    command("analyses/review_completion/scripts/patient_intervals.py"),
    command("analyses/motif_extension/scripts/parity_power.py"),
    command("analyses/motif_extension/scripts/parity_permutation.py"),
]
STAGES = {
    "checks": CHECKS,
    "figures": FIGURES,
    "summaries": SUMMARIES + FIGURES,
    "core": CORE,
    "full": BASELINE
    + SIMULATIONS
    + [[*step, "--rebuild-cache"] if step[0].endswith("paired_power.py") else step for step in CORE]
    + CHECKS,
}


def restore_reference_archives():
    index = json.loads((ROOT / "data/reference_parts/index.json").read_text())
    for entry in index["archives"]:
        target = ROOT / entry["target"]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                digest = hashlib.sha256()
                size = 0
                for part in entry["parts"]:
                    content = (ROOT / part["path"]).read_bytes()
                    if (
                        len(content) != part["size_bytes"]
                        or hashlib.sha256(content).hexdigest() != part["sha256"]
                    ):
                        raise SystemExit(f"Invalid reference archive part: {part['path']}")
                    output.write(content)
                    digest.update(content)
                    size += len(content)
                if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise SystemExit(f"Reference archive checksum mismatch: {entry['target']}")
                output.close()
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Restored reference archive: {entry['target']}")


def verify_integrity():
    manifest = json.loads((ROOT / "MANIFEST_SHA256.json").read_text())
    failures = []
    for name, entry in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f"Missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Changed: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(manifest['files'])} release files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["integrity", *STAGES], default="checks")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="Resume at a numbered step after correcting a failed run.",
    )
    parser.add_argument("--list", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()
    if args.stage == "integrity":
        restore_reference_archives()
        verify_integrity()
        return
    if args.threads < 1 or args.start_step < 1:
        parser.error("threads and start-step must be positive")
    steps = STAGES[args.stage]
    if args.start_step > len(steps):
        parser.error("start-step exceeds the number of steps")
    if args.list:
        for i, step in enumerate(steps, 1):
            print(f"{i:02d}: python {' '.join(step)}")
        return
    restore_reference_archives()
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logdir = ROOT / "reproduction" / f"{args.stage}_{stamp}"
    logdir.mkdir(parents=True)
    for directory in ["figures", "review", "analyses/second_round/cache"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    records = []
    for number, step in enumerate(steps, 1):
        if number < args.start_step:
            continue
        print(f"[{number}/{len(steps)}] {' '.join(step)}", flush=True)
        start = time.perf_counter()
        logfile = logdir / f"{number:02d}_{Path(step[0]).stem}.log"
        with logfile.open("w") as output:
            result = subprocess.run(
                [sys.executable, *step],
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        records.append(
            {
                "step": number,
                "command": step,
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - start,
                "log": str(logfile.relative_to(ROOT)),
            }
        )
        (logdir / "run.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            print(logfile.read_text()[-6000:], file=sys.stderr)
            raise SystemExit(f"Failed at step {number}; see {logfile.relative_to(ROOT)}")
        print(f'  Completed in {records[-1]["seconds"]:.1f} s', flush=True)
    print(f"Completed {args.stage}; logs: {logdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
