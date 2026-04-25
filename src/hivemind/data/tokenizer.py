"""Tokenizer loading indirection.

Wraps ``AutoTokenizer.from_pretrained`` behind a single function so tests
can monkeypatch ``build_tokenizer`` and inject a lightweight whitespace
fake without requiring ``transformers`` or internet at test time.
"""

from __future__ import annotations

from typing import Any, Protocol


class TokenizerProtocol(Protocol):
    """Minimal tokenizer contract used by the collator."""

    pad_token_id: int
    eos_token_id: int | None

    def __call__(
        self,
        text: str | list[str],
        *,
        truncation: bool = ...,
        max_length: int | None = ...,
        padding: bool | str = ...,
        return_tensors: str | None = ...,
    ) -> dict[str, Any]: ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def build_tokenizer(name: str) -> TokenizerProtocol:
    """Load a HuggingFace fast tokenizer and ensure a pad token exists.

    Many causal-LM tokenizers ship without a pad token; we fall back to EOS
    which is the standard convention for left-padded generation and works
    for right-padded training with attention masks.
    """
    from transformers import AutoTokenizer  # lazy — keeps core install minimal

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok
