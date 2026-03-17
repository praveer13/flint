"""Tests for the Flint tokenizer wrapper.

Since transformers is a heavy dependency, tests skip gracefully when it
is not installed.
"""

from __future__ import annotations

import pytest

try:
    from flint_shm.tokenizer import Tokenizer

    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


@pytest.mark.skipif(not HAS_TOKENIZER, reason="transformers not installed")
class TestTokenizer:
    def test_encode_decode_roundtrip(self) -> None:
        tok = Tokenizer("gpt2")  # Small, commonly cached
        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)
        assert "Hello" in text
        assert "world" in text

    def test_single_token_decode(self) -> None:
        tok = Tokenizer("gpt2")
        text = tok.decode_token(tok.encode("Hello")[0])
        assert len(text) > 0

    def test_vocab_size(self) -> None:
        tok = Tokenizer("gpt2")
        assert tok.vocab_size > 0

    def test_eos_token_id(self) -> None:
        tok = Tokenizer("gpt2")
        assert isinstance(tok.eos_token_id, int)

    def test_encode_returns_list_of_ints(self) -> None:
        tok = Tokenizer("gpt2")
        ids = tok.encode("test")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_empty_string(self) -> None:
        tok = Tokenizer("gpt2")
        ids = tok.encode("")
        text = tok.decode(ids)
        assert isinstance(text, str)
