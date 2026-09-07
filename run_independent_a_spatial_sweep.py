#!/usr/bin/env python3
"""Domain-A spatial sweep (6 x 20 epochs), then a fresh 80-epoch selected run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WEIGHTS = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
WARMUP = 4  # Runner uses epoch > warmup: five warm-up epochs, active from epoch index 5.


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select_candidate(rows):
    if len(rows) != len(WEIGHTS) or {r["spatial_weight"] for r in rows} != set(WEIGHTS):
        raise ValueError("selection requires all six distinct completed candidates")
    if not all(math.isfinite(r["validation_foreground"]) for r in rows):
        raise ValueError("non-finite validation metric")
    return max(rows, key=lambda r: (r["validation_foreground"], -r["spatial_weight"]))


def run_training(args, name, weight, epochs, gpu, test):
    output = args.output / name
    command = [
        sys.executable, "-u", "main.py", "--setting-run",
        "--data-root", str(args.data_root), "--sparse-root", str(args.sparse_root),
        "--output", str(output), "--device", "cuda:0", "--seed", "42",
        "--independent-reference", "--independent-task", "1",
        "--method", "zs-sequential", "--epochs-per-task", str(epochs),
        "--batch-size", "4", "--lr", "0.03", "--workers", "8", "--validate-every", "200",
        "--pce-loss-weight", "1", "--zs-global-weight", "1",
        "--zs-spatial-loss-weight", str(weight), "--zs-spatial-warmup-epochs", str(WARMUP),
    ]
    if not test:
        command.append("--independent-skip-test")
    write_json(args.output / (name + ".command.json"), {"gpu": gpu, "command": command})
    start = time.time()
    with (args.output / (name + ".log")).open("x") as log:
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parent,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"},
            stdout=log, stderr=subprocess.STDOUT,
        )
    (args.output / (name + ".exitcode")).write_text(str(result.returncode) + "\n")
    result.check_returncode()
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    record = summary["records"][0]
    assert manifest["status"] == "complete" and summary["complete"]
    assert summary["completed_epochs"] == epochs and summary["iteration"] == 76 * epochs
    assert summary["train_samples"] == 301 and summary["task_order"] == ["A"]
    assert summary["score_split"] == ("test" if test else "val")
    assert summary["test_evaluated"] == test and (record["test"] is not None) == test
    assert manifest["zs_spatial_loss_weight"] == weight
    assert manifest["zs_spatial_warmup_epochs"] == WARMUP
    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    epoch_rows = [r for r in train_rows if "loss" in r]
    active = [r for r in epoch_rows if r["zs_em_mixture_ratios"] is not None]
    assert len(epoch_rows) == epochs
    assert len(active) == (epochs - 5 if weight > 0 else 0)
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["zs_spatial_loss"]) for r in epoch_rows)
    if weight > 0:
        assert active[0]["epoch"] == 5 and any(r["zs_spatial_loss"] > 0 for r in active)
    row = {
        "run": name, "gpu": gpu, "spatial_weight": weight, "epochs": epochs,
        "validation_foreground": record["best_validation"]["foreground_mean"],
        "validation_inclusive": record["best_validation"]["inclusive_mean"],
        "best_epoch": record["best_validation"]["epoch"],
        "best_iteration": record["best_validation"]["iteration"],
        "spatial_active_epochs": len(active),
        "mean_active_spatial_loss": sum(r["zs_spatial_loss"] for r in active) / len(active) if active else 0.0,
        "elapsed_seconds": time.time() - start, "test_evaluated": test,
    }
    if test:
        row["test"] = record["test"]
        row["target_difference"] = record["test"]["foreground_mean"] - 0.7261413306722405
        row["target_reached"] = row["target_difference"] >= 0
    write_json(args.output / (name + ".result.json"), row)
    print(json.dumps(row), flush=True)
    return row


def self_check():
    rows = [{"spatial_weight": w, "validation_foreground": 0.5,
             "test_foreground": 100 * w} for w in WEIGHTS]
    assert select_candidate(rows)["spatial_weight"] == 0  # Ties prefer simpler, never test.
    rows[2]["validation_foreground"] = 0.6
    assert select_candidate(list(reversed(rows)))["spatial_weight"] == 0.05
    try:
        select_candidate(rows[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete sweep must not launch formal training")
    print("selection_self_check_passed")


def main():
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path on experiment storage")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "sweep_running", "scenario": "domain", "task": "A", "seed": 42,
        "spatial_weights": WEIGHTS, "sweep_epochs": 20, "formal_epochs": 80,
        "warmup_epochs": 5, "runner_warmup_argument": WARMUP,
        "pce_weight": 1, "global_weight": 1, "lr": 0.03, "batch_size": 4,
        "workers": 8, "omp_num_threads": 4, "validate_every": 200,
        "optimizer_weight_decay": 0, "manual_gradient_decay": 1e-4,
        "sparse_protocol": args.sparse_root.parent.name, "gpus": [6, 7],
        "selection_metric": "best_foreground_validation_dice",
        "selection_uses_test": False, "formal_initialization": "fresh_seed42",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True,
        ).strip(),
    }
    write_json(args.output / "pipeline.json", protocol)
    try:
        def worker(gpu, indices):
            return [
                run_training(args, f"sweep_s{index:02d}", WEIGHTS[index], 20, gpu, False)
                for index in indices
            ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, 6, range(0, 6, 2)), pool.submit(worker, 7, range(1, 6, 2))]
            rows = [row for future in futures for row in future.result()]
        best = select_candidate(rows)
        write_json(args.output / "sweep_summary.json", {
            "complete": True, "selection_uses_test": False,
            "selection_metric": "best_foreground_validation_dice",
            "candidates": sorted(rows, key=lambda r: r["spatial_weight"]), "best": best,
        })
        protocol.update(status="formal_running", selected_spatial_weight=best["spatial_weight"])
        write_json(args.output / "pipeline.json", protocol)
        formal = run_training(args, "formal80", best["spatial_weight"], 80, 6, True)
        protocol.update(status="complete", formal_result=formal)
        write_json(args.output / "pipeline.json", protocol)
    except Exception as error:
        protocol.update(status="failed", error=repr(error))
        write_json(args.output / "pipeline.json", protocol)
        raise


if __name__ == "__main__":
    main()
