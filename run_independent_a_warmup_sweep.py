#!/usr/bin/env python3
"""Compare 80-epoch Domain-A spatial warmups using validation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from run_independent_a_spatial_sweep import run_training, write_json
from run_independent_domains_formal import gpu_idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-a", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    summary = json.loads((args.completed_a / "summary.json").read_text())
    manifest = json.loads((args.completed_a / "manifest.json").read_text())
    assert summary["complete"] and summary["completed_epochs"] == 80 and summary["task_order"] == ["A"]
    assert manifest["zs_spatial_loss_weight"] == 0.01 and manifest["zs_spatial_warmup_epochs"] == 4
    # Reuse only validation fields from the completed reference for selection.
    validation = summary["records"][0]["best_validation"]
    baseline = {
        "run": "reuse_A_warmup5", "output": str(args.completed_a),
        "warmup_epochs": 5, "runner_warmup_argument": 4,
        "validation_foreground": validation["foreground_mean"],
        "best_epoch": validation["epoch"], "reused": True,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.adopt_running = {}
    protocol = {
        "status": "running", "task": "A", "epochs": 80, "seed": 42,
        "warmup_epochs": [5, 10, 20, 40, 60], "spatial_weight": 0.01,
        "gpus": [4, 5, 6, 7], "selection_uses_test": False,
        "tie_break": "shorter warmup", "baseline": baseline,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(args.output / "pipeline.json", protocol)

    def worker(gpu, warmup_epochs):
        while not gpu_idle(gpu):
            time.sleep(10)
        name = f"warmup{warmup_epochs}"
        print(f"Starting A {name}, first active epoch {warmup_epochs + 1}, GPU {gpu}", flush=True)
        result = run_training(args, name, 0.01, 80, gpu, False, warmup=warmup_epochs - 1)
        return {**result, "warmup_epochs": warmup_epochs, "output": str(args.output / name), "reused": False}

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, gpu, warmup) for gpu, warmup in zip((4, 5, 6, 7), (10, 20, 40, 60))]
            rows = [baseline] + [future.result() for future in futures]
        best = max(rows, key=lambda row: (row["validation_foreground"], -row["warmup_epochs"]))
        write_json(args.output / "selection.json", {"complete": True, "selection_uses_test": False, "candidates": rows, "best": best})
        protocol.update(status="complete", best=best)
    except Exception as error:
        protocol.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(args.output / "pipeline.json", protocol)


if __name__ == "__main__":
    main()
