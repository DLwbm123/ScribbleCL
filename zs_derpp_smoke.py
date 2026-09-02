#!/usr/bin/env python3
"""Static gate for DER++ feature, sparse-PCE, and replay-global state."""
from __future__ import annotations

import json

import numpy as np
import torch

from zs_cl_methods import DarkExperienceReplayPlus


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.head = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward_logits(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return self.head(self.backbone(image))


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    model = Tiny()
    memory = DarkExperienceReplayPlus(buffer_size=5, minibatch_size=3, alpha=0.5, beta=0.5)
    examples = torch.randn(12, 1, 8, 8)
    labels = torch.full((12, 8, 8), -100, dtype=torch.int64)
    labels[:, 2:4, 2:4] = 1
    labels[:, 5:6, 5:7] = 0
    task_ids = torch.zeros(12, dtype=torch.int64)
    with torch.no_grad():
        features = model.backbone(examples)
    memory.add_data(examples, features, labels, task_ids, 2)
    feature_loss, replay = memory.feature_penalty(model, torch.device("cpu"))
    assert replay is not None
    replay_examples, _, replay_labels, replay_tasks, replay_classes = replay
    probabilities = model.forward_logits(replay_examples).softmax(dim=1)
    known = replay_labels.ne(-100)
    pce_loss = -probabilities.gather(1, replay_labels.long().clamp_min(0).unsqueeze(1)).squeeze(1)[known].log().mean()
    total = feature_loss + memory.beta * pce_loss
    total.backward()
    finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    state = memory.state_dict()
    passed = (
        len(memory) == 5
        and memory.num_seen_examples == 12
        and feature_loss < 1e-10
        and float(pce_loss) > 0
        and finite
        and state["examples"].shape[0] == 5
        and state["sparse_labels"].shape == (5, 8, 8)
        and state["task_ids"].shape == (5,)
        and state["class_counts"].shape == (5,)
        and replay_tasks.shape == (3,)
        and replay_classes.shape == (3,)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "stored_examples": len(memory),
        "seen_examples": memory.num_seen_examples,
        "feature_loss_exact_target": float(feature_loss.detach()),
        "replay_pce_loss": float(pce_loss.detach()),
        "finite_gradients": finite,
        "state_bytes": memory.nbytes(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
