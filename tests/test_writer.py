"""Tests for write executor."""

import torch

from hivemind.controller.config import WriterConfig
from hivemind.controller.module_index import ModuleId, ModuleInfo, build_module_index
from hivemind.controller.policy import StoreAction
from hivemind.controller.writer import WriteExecutor
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.stores.fast import FastStore
from hivemind.stores.permanent import PermanentStore
from hivemind.stores.retrieval import RetrievalStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _setup():
    """Create test model, optimizer, and stores."""
    torch.manual_seed(42)
    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    model = StudentModel(config)
    optimizer = MaskedAdamW(model.parameters(), lr=1e-3)
    r_store = RetrievalStore(max_size=100)
    f_store = FastStore()
    p_store = PermanentStore()
    writer = WriteExecutor(model, optimizer, r_store, f_store, p_store)
    modules = build_module_index(model)
    module_map = {m.id: m for m in modules}
    return model, optimizer, writer, r_store, module_map


def test_r_store_adds_to_buffer():
    """R-action adds entry to retrieval store."""
    model, optimizer, writer, r_store, module_map = _setup()

    mid = ModuleId(0, "attn", "F")
    action = StoreAction(module_id=mid, store="R")
    embedding = torch.randn(32)

    assert r_store.size == 0
    writer.execute([action], module_map, step=0, embedding=embedding, bucket_id=0)
    assert r_store.size == 1


def test_r_store_zeros_gradients():
    """R-action zeros the module's gradients."""
    model, optimizer, writer, r_store, module_map = _setup()

    # Create gradients via backward
    tokens = torch.randint(0, 64, (2, 8))
    loss = model(tokens).sum()
    loss.backward()

    mid = ModuleId(0, "attn", "F")
    mod = module_map[mid]

    # Verify gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.params)
    assert has_grad

    action = StoreAction(module_id=mid, store="R")
    writer.execute([action], module_map, step=0, embedding=torch.randn(32))

    # Gradients should be zeroed
    for p in mod.params:
        if p.grad is not None:
            assert p.grad.abs().sum() == 0.0


def test_write_metrics():
    """Execute returns correct count metrics."""
    model, optimizer, writer, r_store, module_map = _setup()

    actions = [
        StoreAction(module_id=ModuleId(0, "attn", "F"), store="R"),
        StoreAction(module_id=ModuleId(0, "ffn", "F"), store="F"),
        StoreAction(module_id=ModuleId(0, "attn", "P"), store="P"),
    ]

    # Need gradients for F-store to work
    tokens = torch.randint(0, 64, (2, 8))
    loss = model(tokens).sum()
    loss.backward()

    metrics = writer.execute(actions, module_map, step=0, embedding=torch.randn(32))
    assert metrics["r_count"] == 1
    assert metrics["f_count"] == 1
    assert metrics["p_count"] == 1
