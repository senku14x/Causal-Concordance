"""Shared fixtures. GPU tests (marker `gpu`) load the pinned model once per session."""
import os
import sys
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMMON))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs the pinned model on a GPU (slow)")


@pytest.fixture(scope="session")
def tok():
    import registry
    return registry.make_tokenizer()


@pytest.fixture(scope="session")
def model():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    import registry
    return registry.load_model()


@pytest.fixture(scope="session")
def brew_bank():
    import wsb
    return wsb.load_bank("brew")[1]


@pytest.fixture(scope="session")
def chain_bank():
    import wsb
    return wsb.load_bank("chain")[1]


@pytest.fixture(scope="session")
def rel_bank():
    import wsb
    return wsb.load_bank("relational")[1]
