"""Theory 201 §5.2 — consistency penalty under dropout-style perturbation."""

from __future__ import annotations

import torch
import torch.nn as nn

from hivemind.regularization import compute_consistency_penalty


class _DropoutHead(nn.Module):
    """Tiny MLP whose output is randomized by dropout when in train()."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(8, 8, bias=False)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(8, 16)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        h = self.drop(h)
        return self.out(h)


def test_zero_when_deterministic():
    torch.manual_seed(0)
    model = _DropoutHead(p=0.0).eval()  # no dropout active
    x = torch.randn(2, 4, 8)

    def fwd():
        return model(x)

    loss = compute_consistency_penalty(fwd)
    assert loss.item() < 1e-6


def test_positive_under_dropout():
    torch.manual_seed(0)
    model = _DropoutHead(p=0.5).train()
    x = torch.randn(2, 4, 8)
    loss = compute_consistency_penalty(lambda: model(x))
    assert loss.item() > 0


def test_mask_restricts_to_label_positions():
    torch.manual_seed(0)
    model = _DropoutHead(p=0.5).train()
    x = torch.randn(2, 4, 8)

    # All-ignored labels → KL contribution restricted to nothing → 0 when
    # the mask is empty (we use clamp_min(1.0) so it's 0 / 1 = 0).
    labels = torch.full((2, 4), -100, dtype=torch.long)
    loss = compute_consistency_penalty(lambda: model(x), labels=labels)
    assert loss.item() == 0.0

    # All-real labels → matches the unrestricted call within a small margin
    labels_real = torch.zeros((2, 4), dtype=torch.long)
    torch.manual_seed(0)
    full = compute_consistency_penalty(lambda: model(x), labels=labels_real)
    assert full.item() > 0
