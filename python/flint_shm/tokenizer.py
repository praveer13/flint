"""Thin wrapper around HuggingFace tokenizers for encode/decode operations.

In Phase 4, tokenization happens on the Python side -- the Zig server
sends raw text to the worker, which tokenizes before running inference.
A future optimization may move tokenization to Zig via C FFI.
"""

from __future__ import annotations

from transformers import AutoTokenizer


class Tokenizer:
    """Wraps a HuggingFace tokenizer for encode/decode operations."""

    def __init__(self, model_name_or_path: str) -> None:
        """Load tokenizer from a HuggingFace model name or local path.

        Examples:
            Tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            Tokenizer("/path/to/local/model")
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its text representation."""
        return self._tokenizer.decode([token_id], skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID."""
        return self._tokenizer.eos_token_id
