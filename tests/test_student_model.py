"""Tests for the student model."""

import torch

from hivemind.student.model import StudentModel


def test_forward_shape(tiny_model, tiny_tokens):
    """Forward produces [B, T, V] logits."""
    logits = tiny_model(tiny_tokens)
    B, T = tiny_tokens.shape
    V = tiny_model.config.vocab_size
    assert logits.shape == (B, T, V)


def test_embed_for_routing_shape(tiny_model, tiny_tokens):
    """Routing embedding produces [B, dim]."""
    h = tiny_model.embed_for_routing(tiny_tokens)
    assert h.shape == (tiny_tokens.shape[0], tiny_model.config.dim)


def test_param_groups(tiny_model):
    """Parameter groups correctly separate P, F, and shared params."""
    groups = tiny_model.get_param_groups()
    assert "P" in groups
    assert "F" in groups
    assert "shared" in groups

    # F should have LoRA params (A and B for each LoRA module)
    assert len(groups["F"]) > 0
    # P should have base weights
    assert len(groups["P"]) > 0
    # Shared should have embedding, norms, etc.
    assert len(groups["shared"]) > 0

    # No overlap
    all_ids = set()
    for group_name, params in groups.items():
        for p in params:
            pid = id(p)
            assert pid not in all_ids, f"Param overlap in group {group_name}"
            all_ids.add(pid)

    # Total should match
    total_params = sum(len(g) for g in groups.values())
    model_params = sum(1 for _ in tiny_model.parameters())
    assert total_params == model_params


def test_module_index_count(tiny_model):
    """Module index enumerates 4L modules (2 layers x 2 blocks x 2 types)."""
    from hivemind.controller.module_index import build_module_index

    modules = build_module_index(tiny_model)
    # 2 layers × (attn_P, attn_F, ffn_P, ffn_F) = 8 modules
    assert len(modules) == 8


def test_merge_all_lora(tiny_student_config):
    """merge_all_lora changes base weights and resets LoRA."""
    torch.manual_seed(42)
    model = StudentModel(tiny_student_config)

    # Set LoRA B to nonzero
    from hivemind.student.lora import LoRALinear

    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.1)

    model.merge_all_lora()

    # After merge, all B matrices should be zero
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert torch.allclose(m.lora_B, torch.zeros_like(m.lora_B))


def test_weight_tying(tiny_model):
    """LM head weight should be tied to embedding weight."""
    assert tiny_model.lm_head.weight is tiny_model.embed.weight
