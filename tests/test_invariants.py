"""
Direct tests of the non-negotiable invariants in CLAUDE.md Section 2.

These do not lock in measured numbers -- they assert the structural properties
the paper's central claim depends on. A change that violates one of these is a
regression regardless of what it does to any metric.
"""
import ast
import os

import pytest

from ingest.metadata_db import build_seed_db
from ingest.normalize import extract_citations
from verify.verifier import CitationVerifier, Chunk, Violation
from eval.verifier_probes import (PROBES, QUOTE_PROBES, REPAIR_PROBES,
                                  SUPPORT_PROBES, SUPPORT_HARD, CONTEXT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY_DIR = os.path.join(ROOT, "verify")

ALL_PROBES = PROBES + QUOTE_PROBES + REPAIR_PROBES

# Anything capable of inference, learned similarity, or a network call. The
# verifier's entire value proposition is that it cannot hallucinate because it
# performs no inference; importing any of these would silently void that.
FORBIDDEN_IMPORTS = {
    "torch", "tensorflow", "keras", "jax", "flax",
    "transformers", "sentence_transformers", "sklearn", "scipy",
    "gensim", "spacy", "faiss", "annoy", "hnswlib", "chromadb",
    "openai", "anthropic", "cohere", "google", "litellm", "ollama",
    "langchain", "llama_index", "haystack",
    "requests", "httpx", "aiohttp", "urllib", "urllib2", "urllib3",
    "http", "socket", "ftplib", "telnetlib", "xmlrpc",
}


@pytest.fixture(scope="module")
def db():
    return build_seed_db()


@pytest.fixture(scope="module")
def chunks(db):
    return [Chunk(i, k, db.get(k).text, db.get(k).verbatim) for i, k in CONTEXT]


def _verify_modules():
    return [os.path.join(VERIFY_DIR, f) for f in sorted(os.listdir(VERIFY_DIR))
            if f.endswith(".py")]


# --------------------------------------------------------------------------
# INVARIANT 1: no model inside verify/
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", _verify_modules(), ids=os.path.basename)
def test_no_model_or_network_imports_in_verify(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

    offending = imported & FORBIDDEN_IMPORTS
    assert not offending, (
        f"{os.path.basename(path)} imports {sorted(offending)}. The verifier "
        f"must perform no inference (CLAUDE.md invariant 1). If semantic "
        f"judgement is needed it belongs in retrieval or generation, described "
        f"in the paper as a weaker guarantee.")


def test_verification_is_deterministic(db, chunks):
    # No inference means no sampling: identical input must give byte-identical
    # output, across separate verifier instances and separate DB builds.
    v1 = CitationVerifier(db, repair=True, support=True)
    v2 = CitationVerifier(build_seed_db(), repair=True, support=True)
    for pid, ans, *_ in ALL_PROBES:
        a, ra = v1.verify(ans, chunks)
        b, rb = v2.verify(ans, chunks)
        assert a == b, f"{pid}: non-deterministic output"
        assert ([f.violation for f in ra.findings]
                == [f.violation for f in rb.findings]), f"{pid}: findings differ"


# --------------------------------------------------------------------------
# INVARIANT 2: citation precision on emitted output is 1.0 BY CONSTRUCTION
# --------------------------------------------------------------------------

def _emitted_body(answer):
    """The claim text only -- annotations are appended after a blank line."""
    return answer.split("\n\n")[0]


@pytest.mark.parametrize("repair_on", [False, True], ids=["strip-only", "repair"])
@pytest.mark.parametrize("support_on", [False, True], ids=["no-v5", "v5"])
def test_emitted_answers_contain_no_unresolvable_citation(
        db, chunks, repair_on, support_on):
    """
    The core guarantee, re-derived from the emitted text alone under every
    configuration -- not read off the verifier's own report.
    """
    import re
    ver = CitationVerifier(db, repair=repair_on, support=support_on)
    valid = {c.index for c in chunks}
    for pid, ans, *_ in ALL_PROBES:
        out, _ = ver.verify(ans, chunks)
        body = _emitted_body(out)
        for n in re.findall(r"\[(\d+)\]", body):
            assert int(n) in valid, f"{pid}: emitted unresolved marker [{n}]"
        for c in extract_citations(body):
            assert db.exists(c.key), f"{pid}: emitted nonexistent {c.key}"


def test_v1_v2_v4_strip_rather_than_annotate(db, chunks):
    """
    V1/V2/V4 must STRIP. If any of them were softened to annotate-only, a
    faulty claim would survive in the emitted body -- which is exactly the
    guarantee the paper claims is structural.
    """
    ver = CitationVerifier(db, repair=False)
    for pid, ans, has_fault, ftypes, note in PROBES:
        out, rep = ver.verify(ans, chunks)
        strips = [f for f in rep.findings
                  if f.violation in (Violation.UNRESOLVED_MARKER,
                                     Violation.NONEXISTENT_PROVISION,
                                     Violation.UNSUPPORTED_QUOTE)]
        if strips:
            assert rep.sentences_stripped > 0, (
                f"{pid}: a V1/V2/V4 violation was reported but nothing was "
                f"stripped -- the check has become annotate-only")


# --------------------------------------------------------------------------
# INVARIANT 4: V3 and V5 annotate, never strip
# --------------------------------------------------------------------------

def test_v3_annotates_without_stripping(db, chunks):
    # A repealed-but-real citation is still good law history: it is annotated
    # with its successor and kept.
    ver = CitationVerifier(db, repair=False)
    for pid, ans, has_fault, ftypes, note in PROBES:
        if ftypes != ["F3"]:
            continue
        out, rep = ver.verify(ans, chunks)
        assert rep.coverage == 1.0, f"{pid}: V3 stripped content"
        assert rep.annotations, f"{pid}: V3 stripped nothing but also said nothing"


def test_v5_never_strips(db):
    """
    Invariant 4, tested where it matters: on claims V5 actively flags. Every
    support probe passes V1-V4 by construction, so any stripping observed here
    could only have come from V5.
    """
    ver = CitationVerifier(db, support=True)
    flagged_any = False
    for pid, claim, key, bad, note in list(SUPPORT_PROBES) + list(SUPPORT_HARD):
        prov = db.get(key)
        ch = [Chunk(1, key, prov.text, prov.verbatim)]
        out, rep = ver.verify(claim, ch)
        if rep.by_type(Violation.WEAK_SUPPORT):
            flagged_any = True
            assert rep.coverage == 1.0, f"{pid}: V5 stripped a claim"
            assert rep.sentences_stripped == 0, f"{pid}: V5 stripped a claim"
    # Class-B guard: the assertion above is vacuous if V5 never fired.
    assert flagged_any, "V5 flagged nothing; the invariant went untested"


# --------------------------------------------------------------------------
# INVARIANT 5: coverage must be reported, and must be non-degenerate
# --------------------------------------------------------------------------

def test_coverage_is_reported_and_bounded(db, chunks):
    ver = CitationVerifier(db, repair=True, support=True)
    for pid, ans, *_ in ALL_PROBES:
        _, rep = ver.verify(ans, chunks)
        assert 0.0 <= rep.coverage <= 1.0, f"{pid}: coverage out of range"
        assert rep.sentences_total > 0, f"{pid}: no sentences counted"


def test_clean_answers_survive_intact(db, chunks):
    ver = CitationVerifier(db, repair=True, support=True)
    for pid, ans, has_fault, ftypes, note in PROBES:
        if has_fault:
            continue
        out, rep = ver.verify(ans, chunks)
        assert rep.coverage == 1.0, f"{pid}: clean answer was stripped"


# --------------------------------------------------------------------------
# GUARDRAILS: CLAUDE.md Section 9
# --------------------------------------------------------------------------

def test_corpus_is_still_the_documented_seed_corpus(db):
    # data/seed_corpus.py is a PIPELINE-BEHAVIOUR corpus, not real law: 35
    # hand-typed provisions with paraphrased text, of which Article 21 alone is
    # verbatim. Every V4 result depends on that one entry, so a change here
    # silently changes what the quotation check was measured against.
    assert db.count() == 35
    verbatim = [p.key for p in db.all_provisions() if p.verbatim]
    assert verbatim == ["CONST§21"], (
        "the set of verbatim provisions changed; V4's measurements were taken "
        "against Article 21 alone and must be re-run")


def test_recodification_mappings_are_still_flagged_unverified(db):
    """
    No MAPPING entry may be flagged verified=True without a citation to the
    authoritative India Code correspondence table it was checked against
    (CLAUDE.md Section 9). The verifier surfaces this provisional status in its
    annotations, so flipping a flag silently changes user-visible output.

    When entries are genuinely verified against an official source, update this
    test alongside the flags and record the source -- do not simply delete it.
    """
    from data.seed_corpus import MAPPING
    unverified = [(o, n) for o, n, v in MAPPING if not v]
    assert len(unverified) == len(MAPPING) == 11, (
        "a recodification mapping was marked verified=True; confirm it was "
        "checked against the official correspondence schedule and cite the "
        "source before relaxing this assertion")


def test_every_mapping_endpoint_exists_in_the_corpus(db):
    # R2 rewrites a superseded reference to its successor. If a successor were
    # absent from the corpus the rewrite would emit a citation that V2 rejects.
    from data.seed_corpus import MAPPING
    for old, new, _ in MAPPING:
        assert db.exists(old), f"mapping source {old} missing from corpus"
        assert db.exists(new), f"mapping target {new} missing from corpus"
