#!/usr/bin/env python3
"""Four 20-epoch demo candidates per domain, then six fresh 80-epoch selected runs."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from run_independent_a_spatial_sweep import run_training, write_json
from run_independent_domains_formal import run_gpu_queue

CANDIDATES = (
    {"lr": 0.03, "global_weight": 1.0},
    {"lr": 0.01, "global_weight": 1.0},
    {"lr": 0.03, "global_weight": 0.1},
    {"lr": 0.01, "global_weight": 0.1},
)
BATCHES = {"A": 76, "B": 42, "C": 87, "D": 42, "E": 146, "F": 176}


def select(rows):
    if len(rows) != 4 or {row["candidate"] for row in rows} != set(range(4)):
        raise ValueError("Exactly four distinct candidates required per domain")
    if len({row["domain"] for row in rows}) != 1:
        raise ValueError("Cannot mix domains in selection")
    if any(row["status"] != "complete" or not math.isfinite(row["demo_selection_foreground"]) for row in rows):
        raise ValueError("Failed or invalid candidates cannot launch formal training")
    return max(rows, key=lambda row: (row["demo_selection_foreground"], -row["candidate"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    args.output.mkdir(parents=True, exist_ok=False)
    args.adopt_running = {}
    protocol = {
        "status": "sweep_running", "domains": list("ABCDEF"), "candidates": CANDIDATES,
        "sweep_epochs": 20, "formal_epochs": 80, "max_sweep_runs_per_domain": 4,
        "seed": 42, "pce_weight": 1.0, "spatial_weight": 0.01, "spatial_first_epoch_index": 10,
        "evaluation_interval": "one epoch", "batches_per_epoch": BATCHES,
        "selection_split": "test", "selection_uses_test": True, "result_usage": "platform_demo",
        "held_out_test_evaluated": False, "formal_initialization": "fresh_seed42",
        "tie_break": "lower candidate index", "gpus": [4, 5, 6, 7],
        "min_free_memory_mib": 12288, "allow_gpu_sharing": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(args.output / "pipeline.json", protocol)

    def execute(job, gpu):
        task, candidate, epochs = job
        domain = "ABCDEF"[task - 1]
        name = f"{domain}_s{candidate}" if epochs == 20 else f"{domain}_formal80"
        print(f"Starting {name} on GPU {gpu}: {CANDIDATES[candidate]}", flush=True)
        try:
            row = run_training(args, name, 0.01, epochs, gpu, True, task=task, warmup=9,
                               demo_test_selection=True, validate_every=BATCHES[domain],
                               **CANDIDATES[candidate])
            row.update(domain=domain, candidate=candidate, status="complete")
        except Exception as error:
            row = {"domain": domain, "candidate": candidate, "status": "failed", "error": repr(error)}
        write_json(args.output / (name + ".result.json"), row)
        return row

    try:
        # Round-robin domains avoids making small domains wait for all large-domain candidates.
        jobs = [(task, candidate, 20) for candidate in range(4) for task in range(1, 7)]
        rows = run_gpu_queue(jobs, execute)
        write_json(args.output / "sweep_results.json", rows)
        selected = {domain: select([row for row in rows if row["domain"] == domain]) for domain in "ABCDEF"}
        write_json(args.output / "selection.json", {
            "complete": True, "selection_split": "test", "result_usage": "platform_demo", "selected": selected,
        })
        protocol.update(status="formal_running", selected=selected)
        write_json(args.output / "pipeline.json", protocol)
        jobs = [(task, selected[domain]["candidate"], 80) for task, domain in enumerate("ABCDEF", 1)]
        formal = run_gpu_queue(jobs, execute)
        protocol["formal_results"] = formal
        if len(formal) != 6 or any(row["status"] != "complete" for row in formal):
            raise RuntimeError("One or more formal runs failed")
        protocol["status"] = "complete"
    except Exception as error:
        protocol.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(args.output / "pipeline.json", protocol)


def self_check():
    from contextlib import redirect_stdout
    import io
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    module = sys.modules[__name__]
    for fail in (False, True):
        calls = []

        def train(args, name, weight, epochs, gpu, test, **kwargs):
            task = kwargs["task"]
            candidate = CANDIDATES.index({key: kwargs[key] for key in ("lr", "global_weight")})
            assert weight == 0.01 and kwargs["warmup"] == 9 and test and kwargs["demo_test_selection"]
            assert kwargs["validate_every"] == BATCHES["ABCDEF"[task - 1]]
            if epochs == 80:
                assert sum(call[2] == 20 for call in calls) == 24
                assert candidate == (task - 1) % 4
            calls.append((task, candidate, epochs))
            if fail and task == 3 and candidate == 2:
                raise RuntimeError("expected candidate failure")
            return {"demo_selection_foreground": float(candidate == (task - 1) % 4), "epochs": epochs}

        with TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            argv = [__file__, "--data-root", directory, "--sparse-root", directory, "--output", str(out)]
            with patch.object(module, "run_training", train), patch.object(module, "run_gpu_queue", side_effect=lambda jobs, fn: [fn(job, 4) for job in jobs]), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                try:
                    main()
                except ValueError:
                    assert fail
                else:
                    assert not fail
            assert len(calls) == (24 if fail else 30)
            assert all(sum(task == domain and epochs == 20 for task, _, epochs in calls) == 4 for domain in range(1, 7))
            assert json.loads((out / "pipeline.json").read_text())["status"] == ("failed" if fail else "complete")
    print("Four-candidate limit, per-domain selection, formal forwarding and failure gate: PASS")


if __name__ == "__main__":
    self_check() if sys.argv[1:] == ["--self-check"] else main()
