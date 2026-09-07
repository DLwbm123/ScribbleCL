#!/usr/bin/env python3
"""Two-batch integration check for explicitly authorized demo test selection."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sparse-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([
        sys.executable, "main.py", "--setting-run", "--data-root", args.data_root,
        "--sparse-root", args.sparse_root, "--output", str(args.output),
        "--device", "cuda:0", "--seed", "42", "--independent-reference", "--independent-task", "1",
        "--method", "zs-sequential", "--epochs-per-task", "2", "--max-train-batches", "1",
        "--batch-size", "4", "--workers", "0", "--validate-every", "1",
        "--zs-spatial-loss-weight", "0.01", "--zs-spatial-warmup-epochs", "0",
        "--independent-demo-test-selection",
    ], check=True)
    summary = json.loads((args.output / "summary.json").read_text())
    manifest = json.loads((args.output / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.output / "train.jsonl").read_text().splitlines()]
    evaluations = [row for row in rows if "validation" in row]
    assert summary["complete"] and summary["iteration"] == 2
    assert summary["result_usage"] == "platform_demo" and summary["test_for_selection"]
    assert not summary["held_out_test_evaluated"] and manifest["selection_split"] == "test"
    assert all(row["selection_split"] == "test" for row in evaluations)
    # Domain A test has three cases; the original validation split has two.
    assert all(len(row["validation"]["per_patient"]) == 3 for row in evaluations)
    best = summary["records"][0]["best_selection"]
    assert summary["records"][0]["test"] is None
    assert "best_validation" not in summary["records"][0]
    assert math.isclose(best["foreground_mean"], max(row["validation"]["foreground_mean"] for row in evaluations))
    assert summary["scores"] == [best["foreground_mean"]]
    assert [row["epoch"] for row in rows if row.get("zs_em_mixture_ratios") is not None] == [1]
    assert (args.output / "best.pt").is_file() and (args.output / "last.pt").is_file()
    print("Demo original-test selection, best checkpoint and labeling: PASS")


if __name__ == "__main__":
    main()
