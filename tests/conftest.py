"""Shared fixtures for the chemberta4 test suite."""

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def load_tokenizer():
    """
    This function is to load the OLMo tokenizer (GPTNeoXTokenizerFast) 
    just once per test session.
    Without this, pytest would reload the tokenizer per test function 
    (default scope), which would be slow and waste memory.
    """
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")
    tok.pad_token = tok.eos_token
    return tok
