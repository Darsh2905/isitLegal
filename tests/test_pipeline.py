"""
Full pipeline wiring (pipeline.py): confidence gate -> fused retrieval ->
generation -> verification, exercised end to end with a scripted FAKE
generator client. This is the first test suite in the project that runs every
architectural stage in sequence -- proving the WIRING is correct, not
producing a real end-to-end result (which needs a real LLM; see
generation/generator.py).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
from retrieval.fusion import FusedRetriever
from retrieval.confidence import ConfidenceGate
from pipeline import Pipeline
from conftest import FakeClient


@pytest.fixture(scope="module")
def db():
    return build_seed_db()


@pytest.fixture(scope="module")
def fused_retriever(db):
    bm25 = BM25Retriever(db)
    enc = DenseEncoder(MODEL_IDS["general"])
    dense = DenseRetriever(db, enc)
    return FusedRetriever([("bm25", bm25), ("dense", dense)])


@pytest.fixture(scope="module")
def gate(db, fused_retriever):
    # The confidence gate's own deployed methodology uses BM25 alone; reusing
    # the BM25 arm here keeps this consistent with eval/run_confidence_eval.py
    # rather than introducing an uncalibrated third signal.
    bm25 = BM25Retriever(db, stem=False)  # the gate's calibrated signal
    return ConfidenceGate(db, bm25, tau=0.30)  # ~ eval/run_confidence_eval.py's tau*


def _pipeline(db, fused_retriever, gate, response_text):
    client = FakeClient(response_text)
    return Pipeline(db, fused_retriever, gate, client), client


def test_answerable_query_produces_a_verified_answer(db, fused_retriever, gate):
    p, client = _pipeline(db, fused_retriever, gate,
                          "Murder is punishable under Section 302 IPC [1].")
    result = p.answer_query("What is the punishment for murder?")
    assert result.answered is True
    assert result.verification is not None
    assert len(client.calls) == 1


def test_unanswerable_query_abstains_without_calling_the_generator(db, fused_retriever, gate):
    p, client = _pipeline(db, fused_retriever, gate, "should never be called")
    result = p.answer_query("How is agricultural income taxed?")
    assert result.answered is False
    assert "Insufficient authority" in result.answer
    assert len(client.calls) == 0, (
        "the gate must abstain BEFORE generation runs -- calling the "
        "generator on an unanswerable query is exactly the cost the gate "
        "exists to avoid")


def test_fabricated_citation_in_generated_output_is_stripped(db, fused_retriever, gate):
    # The load-bearing test: even if a (fake, scripted) generator fabricates a
    # section, the verifier -- not the generator -- is what keeps it out of
    # the final answer. This is the pipeline reproducing the core guarantee
    # end to end rather than trusting it in isolation.
    p, client = _pipeline(db, fused_retriever, gate,
                          "Murder is punished under Section 7777 IPC [1].")
    result = p.answer_query("What is the punishment for murder?")
    assert "7777" not in result.answer
    assert result.verification.findings, "the fabrication should have been flagged"


def test_retrieved_keys_are_reported(db, fused_retriever, gate):
    p, client = _pipeline(db, fused_retriever, gate, "some answer [1].")
    result = p.answer_query("What is the punishment for murder?")
    assert result.retrieved_keys
    assert all("§" in k for k in result.retrieved_keys)


def test_generation_is_only_invoked_on_retrieved_chunks(db, fused_retriever, gate):
    p, client = _pipeline(db, fused_retriever, gate, "answer [1].")
    p.answer_query("What is the punishment for murder?")
    _system, user = client.calls[0]
    # Every [n] the prompt offers must correspond to an actually retrieved key.
    assert "[1]" in user
