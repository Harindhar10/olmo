"""Shared fixtures for the chemberta4 test suite."""

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def tokenizer():
    """Load the OLMo tokenizer (GPTNeoXTokenizerFast) once per test session."""
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")
    tok.pad_token = tok.eos_token
    return tok
