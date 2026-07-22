"""Tests for the write executor's mask-based semantics."""

import torch

from hivemind.controller.module_index import ModuleId, build_module_index
from hivemind.controller.policy import StoreAction
from hivemind.controller.writer import WriteExecutor
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.stores.fast import FastStore
from hivemind.stores.permanent import PermanentStore
from hivemind.stores.retrieval import RetrievalStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _setup(top_k_fraction: float = 0.5):
    """Create test model, optimizer, and stores."""
    torch.manual_seed(42)
    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    model = StudentModel(config)
    optimizer = MaskedAdamW(model.parameters(), lr=1e-3)
    r_store = RetrievalStore(max_size=100)
    f_store = FastStore(top_k_fraction=top_k_fraction)
    p_store = PermanentStore()
    writer = WriteExecutor(model, optimizer, r_store, f_store, p_store)
    modules = build_module_index(model)
    module_map = {m.id: m for m in modules}
    optimizer.register_controller_params(p for m in modules for p in m.params)
    return model, optimizer, writer, r_store, module_map


def _backward(model):
    tokens = torch.randint(0, 64, (2, 8))
    model(tokens).sum().backward()


def test_r_store_adds_to_buffer():
    """R-action adds entry to retrieval store and opens nothing."""
    model, optimizer, writer, r_store, module_map = _setup()

    mid = ModuleId(0, "attn", "F")
    action = StoreAction(module_id=mid, store="R")
    embedding = torch.randn(32)

    assert r_store.size == 0
    metrics, masks = writer.execute(
        [action], module_map, step=0, embedding=embedding, bucket_id=0
    )
    assert r_store.size == 1
    assert masks == {}
    assert metrics["coords_opened"] == 0


def test_r_action_leaves_module_untouched_through_step():
    """R-routed module keeps params AND optimizer state bit-identical."""
    model, optimizer, writer, r_store, module_map = _setup()
    _backward(model)

    mid = ModuleId(0, "attn", "F")
    mod = module_map[mid]
    snapshots = [p.detach().clone() for p in mod.params]

    action = StoreAction(module_id=mid, store="R")
    _, masks = writer.execute([action], module_map, step=0, embedding=torch.randn(32))
    optimizer.set_masks(masks)
    optimizer.step()

    for p, snap in zip(mod.params, snapshots):
        assert torch.equal(p.detach(), snap)
        # State was eagerly initialized at registration and stays all-zero.
        m, v = optimizer.get_state_for_param(p)
        assert m is not None and torch.all(m == 0) and torch.all(v == 0)
        assert float(optimizer.state[p]["coord_step"]) == 0.0


def test_f_action_opens_top_k_only():
    """F-action opens exactly the Top-K rank components of each adapter."""
    model, optimizer, writer, r_store, module_map = _setup(top_k_fraction=0.5)
    _backward(model)

    mid = ModuleId(0, "ffn", "F")
    action = StoreAction(module_id=mid, store="F")
    metrics, masks = writer.execute([action], module_map, step=0)

    assert metrics["f_count"] == 1
    assert len(masks) > 0
    # rank 4, fraction 0.5 -> k=2 components: exactly K/q of each factor opens
    # (paper Proposition 1).
    for param, mask in masks.items():
        assert mask.shape == param.shape
        frac = float(mask.float().sum().item()) / param.numel()
        assert abs(frac - 0.5) < 1e-6
    assert metrics["coords_opened"] == sum(
        int(m.sum().item()) for m in masks.values()
    )


def test_p_action_opens_base_fully():
    """P-action on a P-type module opens every base coordinate."""
    model, optimizer, writer, r_store, module_map = _setup()
    _backward(model)

    mid = ModuleId(0, "attn", "P")
    mod = module_map[mid]
    action = StoreAction(module_id=mid, store="P")
    metrics, masks = writer.execute([action], module_map, step=0)

    assert metrics["p_count"] == 1
    for p in mod.params:
        assert masks[p] is MaskedAdamW.FULLY_OPEN
    assert metrics["coords_opened"] == sum(p.numel() for p in mod.params)


def test_p_on_f_module_opens_adapter_and_flags():
    """P-action on an F-type module opens the whole adapter and flags it."""

    class _StubConsolidator:
        def __init__(self):
            self.flagged = []

        def flag_for_consolidation(self, mid):
            self.flagged.append(mid)

    model, optimizer, writer, r_store, module_map = _setup()
    stub = _StubConsolidator()
    writer.consolidator = stub
    _backward(model)

    mid = ModuleId(0, "ffn", "F")
    mod = module_map[mid]
    action = StoreAction(module_id=mid, store="P")
    _, masks = writer.execute([action], module_map, step=0)

    for p in mod.params:
        assert masks[p] is MaskedAdamW.FULLY_OPEN
    assert stub.flagged == [mid]


def test_write_metrics():
    """Execute returns correct count metrics and per-action detail."""
    model, optimizer, writer, r_store, module_map = _setup()
    _backward(model)

    actions = [
        StoreAction(module_id=ModuleId(0, "attn", "F"), store="R"),
        StoreAction(module_id=ModuleId(0, "ffn", "F"), store="F"),
        StoreAction(module_id=ModuleId(0, "attn", "P"), store="P"),
    ]
    metrics, masks = writer.execute(actions, module_map, step=0, embedding=torch.randn(32))
    assert metrics["r_count"] == 1
    assert metrics["f_count"] == 1
    assert metrics["p_count"] == 1
    assert [a["store"] for a in metrics["actions"]] == ["R", "F", "P"]


def test_unselected_module_invariant_through_masked_steps():
    """A module never selected by the policy stays bit-identical over steps."""
    model, optimizer, writer, r_store, module_map = _setup()

    untouched = module_map[ModuleId(0, "attn", "P")]
    snapshots = [p.detach().clone() for p in untouched.params]

    for step in range(3):
        optimizer.zero_grad()
        _backward(model)
        # Only ever act on the FFN LoRA module.
        action = StoreAction(module_id=ModuleId(0, "ffn", "F"), store="F")
        _, masks = writer.execute([action], module_map, step=step)
        optimizer.set_masks(masks)
        optimizer.step()

    for p, snap in zip(untouched.params, snapshots):
        assert torch.equal(p.detach(), snap)
        m, v = optimizer.get_state_for_param(p)
        assert m is not None and torch.all(m == 0) and torch.all(v == 0)
        assert float(optimizer.state[p]["coord_step"]) == 0.0
