"""
Shared fixtures for the regression suite.

Every test here asserts on numbers returned by the corresponding eval script's
`report()` function -- the same function that prints the human-readable table
used in the paper. Tests therefore cannot drift from the published tables: both
read the same computation.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _report(module_name):
    import importlib
    return importlib.import_module(module_name).report()


@pytest.fixture(scope="session")
def normalizer():
    return _report("eval.run_normalizer_eval")


@pytest.fixture(scope="session")
def verifier():
    return _report("eval.run_verifier_eval")


@pytest.fixture(scope="session")
def quote():
    return _report("eval.run_quote_eval")


@pytest.fixture(scope="session")
def retrieval():
    return _report("eval.run_retrieval_eval")


@pytest.fixture(scope="session")
def inforce():
    return _report("eval.run_inforce_eval")


@pytest.fixture(scope="session")
def repair():
    return _report("eval.run_repair_eval")


@pytest.fixture(scope="session")
def xref():
    return _report("eval.run_xref_eval")


@pytest.fixture(scope="session")
def xref_sweep():
    return _report("eval.run_xref_k_sweep")


@pytest.fixture(scope="session")
def support():
    return _report("eval.run_support_eval")


@pytest.fixture(scope="session")
def support_hard():
    return _report("eval.run_support_hard")


@pytest.fixture(scope="session")
def evasion():
    return _report("eval.run_evasion_eval")


@pytest.fixture(scope="session")
def support_xref():
    return _report("eval.run_support_xref")


@pytest.fixture(scope="session")
def confidence():
    return _report("eval.run_confidence_eval")


@pytest.fixture(scope="session")
def confidence_real():
    return _report("eval.run_confidence_real_eval")


@pytest.fixture(scope="session")
def live_generation():
    return _report("eval.run_live_generation_eval")


@pytest.fixture(scope="session")
def confidence_external():
    return _report("eval.run_confidence_external_eval")


class FakeClient:
    """Scripted stand-in for an LLMClient: returns a pre-written string and
    records what it was called with. Not a model -- a fixed function of its
    script. Shared by test_generator.py and test_pipeline.py."""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


@pytest.fixture(scope="session")
def verifier_real():
    return _report("eval.run_verifier_real_eval")

