#!/usr/bin/env python3
"""Joint-domain ZS training upper bound for Domain-CL."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from zs_setting_runner import (
    TASKS,
    DomainModel,
    H5Slices,
    _evaluate_task,
    _loader,
    _sparse_path,
    evaluate,
    native_target,
    pce_loss,
    zs_cutout_invariance,
)


def validation_scores(model, tasks, loaders, datasets, device) -> dict[str, float]:
    return {
        task.code: evaluate(model, loader, dataset.ends, device, None, task.classes)[
            "benchmark_mean"
        ]
        for task, loader, dataset in zip(tasks, loaders, datasets)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--pce-loss-weight", type=float, default=1.0)
    parser.add_argument("--zs-global-weight", type=float, default=1.0)
    parser.add_argument("--zs-gd-loss", action="store_true")
    parser.add_argument("--zs-adversarial-perturbation", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.validate_every) < 1 or args.lr <= 0:
        parser.error("epochs, batch size, validation interval, and learning rate must be positive")
    if min(args.pce_loss_weight, args.zs_global_weight) < 0:
        parser.error("loss weights must be non-negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be positive")

    tasks = TASKS["domain"]
    for task in tasks:
        if not (args.data_root / task.folder / task.filename).is_file():
            raise FileNotFoundError(f"missing data for domain {task.code}")
        if not _sparse_path(args.sparse_root, "domain", task, args.seed).is_file():
            raise FileNotFoundError(f"missing sparse annotations for domain {task.code}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)

    train_sets = [
        H5Slices(
            args.data_root / task.folder / task.filename,
            "train",
            _sparse_path(args.sparse_root, "domain", task, args.seed),
            augment=True,
        )
        for task in tasks
    ]
    val_sets = [H5Slices(args.data_root / task.folder / task.filename, "val") for task in tasks]
    joint_train = ConcatDataset(train_sets)
    if len(joint_train) != sum(len(dataset) for dataset in train_sets):
        raise RuntimeError("joint dataset coverage mismatch")
    train_loader = _loader(joint_train, args.batch_size, True, args.workers, args.seed)
    val_loaders = [
        _loader(dataset, args.batch_size, False, 0, args.seed) for dataset in val_sets
    ]

    model = DomainModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    batches_per_epoch = len(train_loader)
    if args.max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
    max_iterations = batches_per_epoch * args.epochs
    manifest = {
        "scenario": "domain",
        "method": "zs-joint",
        "mode": "joint_upper_bound",
        "backbone": "ZScribbleSeg_UNet",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "task_order": [task.code for task in tasks],
        "train_samples": len(joint_train),
        "batches_per_epoch": batches_per_epoch,
        "validation_selection": "equal_mean_domain_dice",
        "validate_every": args.validate_every,
        "pce_loss_weight": args.pce_loss_weight,
        "zs_global_weight": args.zs_global_weight,
        "data_root": "<external_data>",
        "sparse_root": "<external_data>",
        "status": "running",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    train_log = args.output / "train.jsonl"
    best_path = args.output / "joint_best.pt"
    best = {"mean_domain_dice": -1.0, "epoch": None, "iteration": None}
    iteration = 0

    def validate(epoch: int) -> None:
        nonlocal best
        model.eval()
        scores = validation_scores(model, tasks, val_loaders, val_sets, device)
        mean_score = float(np.mean(list(scores.values())))
        record = {
            "epoch": epoch,
            "iteration": iteration,
            "validation": {"per_domain": scores, "mean_domain_dice": mean_score},
        }
        with train_log.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if mean_score > best["mean_domain_dice"]:
            best = {"mean_domain_dice": mean_score, "epoch": epoch, "iteration": iteration}
            torch.save(model.state_dict(), best_path)
        model.train()

    try:
        with train_log.open("a") as stream:
            for epoch in range(args.epochs):
                model.train()
                totals = {"loss": [], "pce": [], "global": [], "gd": []}
                for batch_index, (image, label) in enumerate(train_loader):
                    if batch_index >= batches_per_epoch:
                        break
                    image, label = image.to(device), label.to(device)
                    target = native_target(label, 2)
                    optimizer.zero_grad(set_to_none=True)
                    outputs, global_loss, gd_loss, _ = zs_cutout_invariance(
                        model, image, target, None, args, device,
                    )
                    pce = pce_loss(outputs["pred_masks"], target)
                    loss = args.pce_loss_weight * pce + args.zs_global_weight * global_loss
                    if args.zs_gd_loss:
                        loss = loss + gd_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite joint-training loss")
                    loss.backward()
                    optimizer.step()
                    iteration += 1
                    learning_rate = args.lr * max(0.0, 1.0 - iteration / max_iterations) ** 0.9
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    totals["loss"].append(float(loss.detach()))
                    totals["pce"].append(float(pce.detach()))
                    totals["global"].append(float(global_loss.detach()))
                    totals["gd"].append(float(gd_loss.detach()))
                    if iteration % args.validate_every == 0:
                        stream.flush()
                        validate(epoch)
                stream.write(json.dumps({
                    "epoch": epoch,
                    "iteration": iteration,
                    "loss": float(np.mean(totals["loss"])),
                    "pce_loss": float(np.mean(totals["pce"])),
                    "zs_global_loss": float(np.mean(totals["global"])),
                    "zs_gd_loss": float(np.mean(totals["gd"])),
                }, sort_keys=True) + "\n")
                stream.flush()

        validate(args.epochs - 1)
        model.load_state_dict(torch.load(best_path, map_location=device))
        torch.save(model.state_dict(), args.output / "joint_model.pt")
        test_scores = {
            task.code: _evaluate_task(
                model, "domain", task, index, args.data_root, "test", args.batch_size, device,
            )["benchmark_mean"]
            for index, task in enumerate(tasks)
        }
        summary = {
            "method": "zs-joint",
            "best_validation": best,
            "test_per_domain": test_scores,
            "test_mean_domain_dice": float(np.mean(list(test_scores.values()))),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for dataset in (*train_sets, *val_sets):
            dataset.close()


if __name__ == "__main__":
    main()
