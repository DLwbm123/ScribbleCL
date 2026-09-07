#!/usr/bin/env python3
"""Compare the real first optimizer step with the preserved stage-A runner.

Run on the experiment host with --oracle-checkout, --data-root, --sparse-root,
and --output (a new directory on experiment storage). No test images are read.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import runner_core as current


class FirstStepComplete(Exception):
    pass


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def capture(module, arguments, batch):
    captured = {}
    build_model, pce, invariance = module._build_model, module.pce_loss, module.zs_cutout_invariance
    real_loader, real_sgd, backward = module._loader, torch.optim.SGD, torch.Tensor.backward

    def build(scenario):
        model = build_model(scenario)
        captured["initial"] = snapshot(model)
        captured["model"] = model
        return model

    def partial_ce(*args, **kwargs):
        result = pce(*args, **kwargs)
        captured["pce_loss"] = float(result.detach())
        return result

    def global_loss(*args, **kwargs):
        result = invariance(*args, **kwargs)
        captured["global_loss"] = float(result[1].detach())
        return result

    def loss_backward(tensor, *args, **kwargs):
        captured["total_loss"] = float(tensor.detach())
        return backward(tensor, *args, **kwargs)

    class FixedBatch:
        def __len__(self):
            return 76

        def __iter__(self):
            yield tuple(value.clone() for value in batch)

    def loader(dataset, *args):
        return FixedBatch() if dataset.split == "train" else real_loader(dataset, *args)

    class Optimizer(real_sgd):
        def step(self, *args, **kwargs):
            captured["optimizer"] = self
            captured["optimizer_weight_decay"] = self.param_groups[0]["weight_decay"]
            captured["lr_before"] = self.param_groups[0]["lr"]
            captured["gradients"] = {
                name: value.grad.detach().cpu().clone()
                for name, value in captured["model"].named_parameters() if value.grad is not None
            }
            result = super().step(*args, **kwargs)
            captured["updated"] = snapshot(captured["model"])
            return result

    def evaluate(*args, **kwargs):
        if "updated" in captured:
            captured["lr_after"] = captured["optimizer"].param_groups[0]["lr"]
            raise FirstStepComplete
        return {"benchmark_mean": 0.0}

    with ExitStack() as stack:
        for obj, name, value in (
            (module, "_build_model", build), (module, "pce_loss", partial_ce),
            (module, "zs_cutout_invariance", global_loss), (module, "_loader", loader),
            (module, "evaluate", evaluate), (torch.optim, "SGD", Optimizer),
            (torch.Tensor, "backward", loss_backward), (sys, "argv", ["parity", *arguments]),
        ):
            stack.enter_context(patch.object(obj, name, value))
        try:
            module.main("domain")
        except FirstStepComplete:
            pass
        else:
            raise AssertionError("runner did not reach the first scheduled validation")
    del captured["model"], captured["optimizer"]
    torch.cuda.empty_cache()
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Diagnostic only: isolate implementation parity from nondeterministic CUDA kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    spec = importlib.util.spec_from_file_location("stage_a_oracle", args.oracle_checkout / "runner_core.py")
    oracle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = oracle
    spec.loader.exec_module(oracle)
    oracle.TASKS["domain"] = oracle.TASKS["domain"][:1]
    task = oracle.TASKS["domain"][0]
    train = oracle.H5Slices(
        args.data_root / task.folder / task.filename, "train",
        oracle._sparse_path(args.sparse_root, "domain", task, 42), augment=True,
    )
    train_loader = oracle._loader(train, 4, True, 8, 42)
    assert len(train) == 301 and len(train_loader) == 76
    batch = next(iter(train_loader))
    train.close()
    common = [
        "--data-root", str(args.data_root), "--sparse-root", str(args.sparse_root),
        "--device", "cuda:0", "--seed", "42", "--epochs-per-task", "150",
        "--batch-size", "4", "--lr", "0.03", "--workers", "8", "--validate-every", "1",
        "--pce-loss-weight", "1", "--zs-global-weight", "1", "--zs-spatial-loss-weight", "0",
    ]
    reference = capture(oracle, common + [
        "--output", str(args.output / "oracle"), "--method", "zs-gpm", "--max-task", "1",
    ], batch)
    repeat = capture(oracle, common + [
        "--output", str(args.output / "oracle_repeat"), "--method", "zs-gpm", "--max-task", "1",
    ], batch)
    candidate = capture(current, common + [
        "--output", str(args.output / "independent"), "--method", "zs-sequential",
        "--independent-reference", "--independent-task", "1", "--independent-skip-test",
    ], batch)
    report = {"passed": True, "cudnn_deterministic": True, "train_samples": 301,
              "batches_per_epoch": 76, "max_iterations": 11400}
    for key in ("initial", "gradients", "updated"):
        assert reference[key].keys() == candidate[key].keys()
        largest = repeat_largest = 0.0
        for name, value in reference[key].items():
            largest = max(largest, float((value.to(torch.float64) - candidate[key][name]).abs().max()))
            repeat_largest = max(repeat_largest, float((value.to(torch.float64) - repeat[key][name]).abs().max()))
        report[key + "_max_abs_diff"] = largest
        report[key + "_oracle_repeat_max_abs_diff"] = repeat_largest
    for key in ("pce_loss", "global_loss", "total_loss", "lr_before", "lr_after", "optimizer_weight_decay"):
        report[key] = {"oracle": reference[key], "independent": candidate[key]}
    assert candidate["optimizer_weight_decay"] == 0
    assert abs(candidate["lr_after"] - 0.03 * (1 - 1 / 11400) ** 0.9) < 1e-12
    report["manual_gradient_decay"] = 1e-4
    report["scope"] = "one fixed augmented training batch; no test access; real shared loops and optimizer"
    limits = {"initial": 0.0, "gradients": 1e-5, "updated": 1e-6}
    report["absolute_tolerances"] = limits
    report["passed"] = all(report[key + "_max_abs_diff"] <= limit for key, limit in limits.items())
    report["passed"] &= all(abs(reference[key] - candidate[key]) < 1e-7 for key in (
        "pce_loss", "global_loss", "total_loss", "lr_before", "lr_after", "optimizer_weight_decay"))
    (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["passed"], "first-step parity failed; inspect parity.json"


if __name__ == "__main__":
    main()
