#!/usr/bin/env python3
"""Evaluate only the validation-selected warmup checkpoint on Domain-A test."""
import argparse
import json
from pathlib import Path

import torch
from runner_core import TASKS, _build_model, _evaluate_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Selected-test output already exists; do not repeat evaluation")
    selection = json.loads(args.selection.read_text())
    assert selection["complete"] and not selection["selection_uses_test"]
    best = max(selection["candidates"], key=lambda r: (r["validation_foreground"], -r["warmup_epochs"]))
    assert selection["best"] == best
    directory = Path(best["output"])
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["complete"] and summary["task_order"] == ["A"]
    assert summary["completed_epochs"] == 80
    if summary["test_evaluated"]:
        test = summary["records"][0]["test"]
        reused = True
    else:
        device = torch.device("cuda:0")
        model = _build_model("domain").to(device)
        model.activate_stage(0)
        model.load_state_dict(torch.load(directory / "best.pt", map_location=device))
        test = _evaluate_task(model, "domain", TASKS["domain"][0], 0, args.data_root, "test", 4, device)
        test = {**test, "benchmark_mean": test["foreground_mean"], "dice_includes_background": False}
        reused = False
    result = {
        "complete": True, "task": "A", "selected_run": best["run"],
        "warmup_epochs": best["warmup_epochs"], "spatial_weight": 0.01,
        "validation_foreground": best["validation_foreground"],
        "best_epoch": best["best_epoch"], "selection_uses_test": False,
        "test_reused": reused, "test": test,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
