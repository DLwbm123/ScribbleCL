#!/usr/bin/env python3
"""Run independent B-F with Domain A's selected configuration on GPUs 4-7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

from run_independent_a_spatial_sweep import run_training, write_json


def gpu_idle(gpu):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    memory, utilization = map(int, output.strip().split(","))
    return memory < 100 and utilization == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-a", type=Path, required=True)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.warmup_epochs < 80:
        parser.error("--warmup-epochs must be between 0 and 79")
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    prior = json.loads((args.completed_a / "summary.json").read_text())
    assert prior["complete"] and prior["task_order"] == ["A"] and prior["completed_epochs"] == 80
    args.output.mkdir(parents=True, exist_ok=False)
    args.adopt_running = {}
    protocol = {
        "status": "running", "gpus": [4, 5, 6, 7], "domains": list("ABCDEF"),
        "completed_a": str(args.completed_a), "fresh_domains": list("BCDEF"),
        "epochs": 80, "seed": 42, "spatial_weight": 0.01,
        "spatial_first_epoch_index": args.warmup_epochs, "configuration_selected_on": "A validation",
        "selection_uses_test": False, "training_implementation": "shared_stage_loop",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(args.output / "pipeline.json", protocol)
    pending = Queue()
    for task in range(2, 7):
        pending.put(task)

    def worker(gpu):
        rows = []
        while not pending.empty():
            if not gpu_idle(gpu):
                time.sleep(10)
                continue
            try:
                task = pending.get_nowait()
            except Empty:
                break
            domain = "ABCDEF"[task - 1]
            print(f"Starting Domain {domain} on GPU {gpu}", flush=True)
            try:
                row = run_training(args, domain, 0.01, 80, gpu, True, task=task,
                                   warmup=args.warmup_epochs - 1)
                rows.append({"domain": domain, "status": "complete", **row})
            except Exception as error:
                row = {"domain": domain, "gpu": gpu, "status": "failed", "error": repr(error)}
                write_json(args.output / f"{domain}.failure.json", row)
                rows.append(row)
            finally:
                pending.task_done()
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (4, 5, 6, 7)]
        rows = sorted([row for future in futures for row in future.result()], key=lambda row: row["domain"])
    protocol["results"] = rows
    protocol["status"] = "complete" if len(rows) == 5 and all(r["status"] == "complete" for r in rows) else "failed"
    write_json(args.output / "pipeline.json", protocol)
    if protocol["status"] != "complete":
        raise RuntimeError("One or more domains failed; see pipeline.json")


def self_check():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    module = sys.modules[__name__]
    for fail in (False, True):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "summary.json", {"complete": True, "task_order": ["A"], "completed_epochs": 80})
            calls = []

            def train(args, name, weight, epochs, gpu, test, task, warmup):
                assert name == "ABCDEF"[task - 1] and weight == 0.01 and epochs == 80 and test
                assert warmup == 9
                calls.append(task)
                if fail and task == 3:
                    raise RuntimeError("expected test failure")
                return {"gpu": gpu}

            argv = [__file__, "--data-root", directory, "--sparse-root", directory,
                    "--completed-a", directory, "--output", str(root / "output"), "--warmup-epochs", "10"]
            with patch.object(module, "run_training", train), patch.object(module, "gpu_idle", return_value=True), patch.object(sys, "argv", argv):
                try:
                    main()
                except RuntimeError:
                    assert fail
                else:
                    assert not fail
            result = json.loads((root / "output/pipeline.json").read_text())
            assert sorted(calls) == [2, 3, 4, 5, 6]
            assert result["status"] == ("failed" if fail else "complete")
            assert result["spatial_first_epoch_index"] == 10
    print("Domain queue and failure handling: PASS")


if __name__ == "__main__":
    self_check() if sys.argv[1:] == ["--self-check"] else main()
