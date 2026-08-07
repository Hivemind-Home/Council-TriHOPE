"""Tests for phase-resolved retention tracking and gold exact-match probes."""

from __future__ import annotations

import torch

from hivemind.evaluation import (
    GoldProbeSet,
    PhaseEvalTracker,
    build_gold_probes,
    exact_match_eval,
    normalize_answer,
)


class TestPhaseEvalTracker:
    def test_retention_delta_against_own_phase_baseline(self) -> None:
        tr = PhaseEvalTracker()
        # End of general phase: general trained, loss 1.0.
        d1 = tr.record(
            phase="general_warm", step=100,
            phase_domains=["general"],
            domain_loss={"general": 1.0, "code": 3.0},
        )
        assert d1 == {}  # nothing has an own-phase baseline yet

        # End of code phase: code trained; general drifted to 1.4.
        d2 = tr.record(
            phase="code_recurrent", step=300,
            phase_domains=["code"],
            domain_loss={"general": 1.4, "code": 1.1},
        )
        assert d2 == {"general": 1.4 - 1.0}

        # End of math phase: both old domains measured against their own bests.
        d3 = tr.record(
            phase="math_recurrent", step=500,
            phase_domains=["math"],
            domain_loss={"general": 1.6, "code": 1.5, "math": 0.9},
        )
        assert d3["general"] == 1.6 - 1.0
        assert d3["code"] == 1.5 - 1.1

    def test_revisit_refreshes_baseline(self) -> None:
        tr = PhaseEvalTracker()
        tr.record(phase="p1", step=1, phase_domains=["code"], domain_loss={"code": 2.0})
        tr.record(
            phase="p2", step=2, phase_domains=["math"],
            domain_loss={"code": 2.5, "math": 1.0},
        )
        # Revisit code: baseline refreshes to the new end-of-phase loss.
        tr.record(phase="p3", step=3, phase_domains=["code"], domain_loss={"code": 1.8})
        d = tr.record(phase="p4", step=4, phase_domains=["math"], domain_loss={"code": 2.0})
        assert abs(d["code"] - (2.0 - 1.8)) < 1e-12

    def test_state_dict_roundtrip(self) -> None:
        tr = PhaseEvalTracker()
        tr.record(phase="p1", step=1, phase_domains=["a"], domain_loss={"a": 1.0})
        clone = PhaseEvalTracker()
        clone.load_state_dict(tr.state_dict())
        assert clone.retention_table() == tr.retention_table()


class TestNormalizeAnswer:
    def test_boxed_extraction(self) -> None:
        assert normalize_answer(r"The result is \boxed{42}.") == "42"

    def test_last_number(self) -> None:
        assert normalize_answer("First 3, then 5, so x = 7") == "7"
        assert normalize_answer("1,234 apples") == "1234"

    def test_decimal_trailing_zeros(self) -> None:
        assert normalize_answer("x = 2.50") == normalize_answer("2.5")

    def test_text_fallback(self) -> None:
        assert normalize_answer("  Paris  ") == "paris"
        assert normalize_answer("PARIS is the answer") == "paris is the answer"


class _CharTokenizer:
    """Deterministic toy tokenizer: one token per character code."""

    eos_token_id = 0

    def __call__(
        self,
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        # add_special_tokens is accepted (and ignored) to match the real
        # HF tokenizer signature the eval path calls with.
        ids = [min(ord(c), 255) for c in text][:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i > 0)


class _EchoStudent(torch.nn.Module):
    """Predicts a fixed answer string then EOS, regardless of input."""

    def __init__(self, answer: str):
        super().__init__()
        self.answer_ids = [ord(c) for c in answer] + [0]
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        # Position within the generation = tokens beyond the prompt. The
        # harness appends generated tokens, so infer position from calls.
        # Simpler: emit answer based on how many of our answer tokens are
        # already at the end of the sequence.
        seq = tokens[0].tolist()
        pos = 0
        for k in range(min(len(self.answer_ids), len(seq)), 0, -1):
            if seq[-k:] == self.answer_ids[:k]:
                pos = k
                break
        next_id = self.answer_ids[pos] if pos < len(self.answer_ids) else 0
        logits = torch.zeros(1, len(seq), 256)
        logits[0, -1, next_id] = 10.0
        return logits


class TestExactMatchEval:
    def test_correct_answer_scores_one(self) -> None:
        probes = GoldProbeSet(
            domain="math",
            rows=[{"input_text": "What is 3+4?", "target_text": "7"}],
        )
        student = _EchoStudent("x = 7")
        em = exact_match_eval(
            student, _CharTokenizer(), probes, torch.device("cpu"), max_new_tokens=16
        )
        assert em == 1.0

    def test_wrong_answer_scores_zero(self) -> None:
        probes = GoldProbeSet(
            domain="math",
            rows=[{"input_text": "What is 3+4?", "target_text": "8"}],
        )
        student = _EchoStudent("x = 7")
        em = exact_match_eval(
            student, _CharTokenizer(), probes, torch.device("cpu"), max_new_tokens=16
        )
        assert em == 0.0


class _GoldDataset:
    def __init__(self):
        self.rows = [
            {"input_text": f"q{i}", "target_text": f"{i}" if i % 2 == 0 else ""}
            for i in range(10)
        ]

    def gold_indices(self, name):
        if name != "math":
            raise KeyError(name)
        return [i for i in range(10) if i % 2 == 0]

    def __getitem__(self, i):
        return self.rows[i]


class TestBuildGoldProbes:
    def test_deterministic_selection(self) -> None:
        ds = _GoldDataset()
        p1 = build_gold_probes(ds, ["math"], num_samples=3, seed=1)
        p2 = build_gold_probes(ds, ["math"], num_samples=3, seed=1)
        assert [r["input_text"] for r in p1["math"].rows] == [
            r["input_text"] for r in p2["math"].rows
        ]
        assert len(p1["math"].rows) == 3

    def test_missing_domain_skipped(self) -> None:
        probes = build_gold_probes(_GoldDataset(), ["code"], num_samples=3, seed=1)
        assert probes == {}
