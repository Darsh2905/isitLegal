"""
The verifier on the REAL corpus (eval/run_verifier_real_eval.py), plus unit
tests for each mechanism added in the same revision: sub-section resolution,
unknown-act detection, act-year handling, sentence segmentation, elided
quotations, marker/citation consistency, the full correspondence mapping and
in-force preference.

Each mechanism is tested on BOTH the fault it exists to catch and the
legitimate construct it must leave alone (CLAUDE.md Section 6, Class C).
"""
import pytest

from ingest.normalize import extract_citations, extract_unresolved_act_refs
from verify.verifier import _segment, _elided_quote_ok


# --------------------------------------------------------------------------
# Headline: the probe set as a whole
# --------------------------------------------------------------------------

def test_every_fault_is_stripped(verifier_real):
    assert verifier_real["fn"] == 0, [r for r in verifier_real["rows"] if not r["correct"]]
    assert verifier_real["tp"] >= 17


def test_no_correct_citation_is_stripped(verifier_real):
    assert verifier_real["fp"] == 0, [r for r in verifier_real["rows"] if not r["correct"]]
    assert verifier_real["tn"] >= 14


def test_independent_audit_passes_on_every_emitted_answer(verifier_real):
    assert verifier_real["audit_pass"] == verifier_real["n"]


def test_marker_contradictions_are_repointed_or_flagged_never_stripped(verifier_real):
    assert verifier_real["marker_ok"] == verifier_real["marker_n"]


# --------------------------------------------------------------------------
# Sub-section resolution
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_db():
    from ingest.india_code import build_real_db
    return build_real_db()


@pytest.mark.parametrize("key", ["CONST§19(1)(a)", "CONST§19(1)(g)", "CRPC§41(1)(b)",
                                 "BNS§103(1)", "BNS§318(2)", "CONST§22(3)(a)"])
def test_real_pinpoint_resolves_to_its_section(real_db, key):
    assert real_db.resolve_key(key) == key.split("(")[0]


@pytest.mark.parametrize("key", ["IPC§302(7)", "CONST§21(1)", "BNS§103(3)",
                                 "CONST§22(3)(c)", "IPC§9999(1)",
                                 # "(a)" occurs in clause (2) only as a
                                 # cross-reference to sub-clause (a) of clause (1)
                                 "CONST§19(2)(a)"])
def test_fabricated_pinpoint_does_not_resolve(real_db, key):
    assert real_db.resolve_key(key) is None


def test_exact_lookup_is_unchanged(real_db):
    assert real_db.get("CONST§19(1)(a)") is None  # get() stays exact
    assert real_db.resolve_key("IPC§420") == "IPC§420"


# --------------------------------------------------------------------------
# Unknown acts, and act years
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Section 5 of the Indian Cyber Safety Act, 2019 makes phishing punishable.",
    "Under section 118 of the Bharatiya Samvidhan Adhiniyam, a presumption arises.",
    "Section 9 of the Code of Civil Procedure bars such suits.",
    "Section 5 of the indian cyber safety act applies.",
    "Section 12 of the Cyber Security Law applies.",
    "Vide Manipur Act 3 of 1985, Section 4(2), Sch.",
])
def test_section_of_an_act_outside_the_corpus_is_flagged(text):
    assert extract_unresolved_act_refs(text)


@pytest.mark.parametrize("text", [
    "Section 420 of the Indian Penal Code punishes cheating.",
    "Section 41 of the code of criminal procedure permits arrest.",
    "Section 66 of the IT Act 2000 punishes hacking.",
    "Section 3 of the Act applies.", "Section 3 of the said act applies.",
    "Section 3 of this Sanhita applies.", "Section 103 of the new Act applies.",
    "He decided to act under section 41 of the CrPC.",
    "The Indian Evidence Act, Section 65B governs electronic records.",
])
def test_known_acts_and_back_references_are_not_flagged(text):
    assert not extract_unresolved_act_refs(text)


def test_an_acts_own_year_is_not_a_section():
    assert [c.key for c in extract_citations("The IPC 1860 was replaced by the BNS 2023.")] == []
    assert [c.key for c in extract_citations("IT Act 2000 Section 66 applies.")] == ["ITA§66"]


def test_any_other_year_shaped_number_is_still_checked():
    assert [c.key for c in extract_citations("IPC 1999 punishes cyber stalking.")] == ["IPC§1999"]
    assert [c.key for c in extract_citations("IPC §1860 is fake.")] == ["IPC§1860"]


# --------------------------------------------------------------------------
# Segmentation (constructs from real gpt-oss-20b output)
# --------------------------------------------------------------------------

def test_markers_after_the_full_stop_stay_with_their_claim():
    segs = [s for s, _ in _segment("Murder is punishable with death. [1][2] Next claim [3].")]
    assert segs == ["Murder is punishable with death. [1][2]", "Next claim [3]."]


def test_marker_glued_to_the_full_stop_still_ends_the_sentence():
    segs = [s for s, _ in _segment("It is a dowry death.[1]  \n\nThe same rule is in section 80.[2]")]
    assert segs == ["It is a dowry death.[1]", "The same rule is in section 80.[2]"]


def test_bullets_and_numbered_items_are_separate_sentences():
    text = "Safeguards:\n1. Right to counsel [1].\n- Production within 24 hours [2]"
    segs = [s for s, _ in _segment(text)]
    assert segs == ["Safeguards:", "1. Right to counsel [1].", "- Production within 24 hours [2]"]


def test_abbreviations_still_do_not_split():
    segs = [s for s, _ in _segment("The cost was Rs. 53 crores under Sec. 420 IPC [1].")]
    assert len(segs) == 1


# --------------------------------------------------------------------------
# Elided quotations
# --------------------------------------------------------------------------

SRC = ("No person shall be convicted of any offence except for violation of a "
       "law in force, nor be subjected to a penalty greater than that which might "
       "have been inflicted. Any person who for the time being is an enemy alien.")


def test_harmless_elision_is_accepted_and_the_omission_is_reported():
    ok, omitted = _elided_quote_ok("Any person who…is an enemy alien", SRC)
    assert ok and omitted == ["for the time being"]


@pytest.mark.parametrize("quote", [
    # omits "except ... nor": an exception and a negation
    "No person shall be convicted of any offence … be subjected to a penalty greater",
    # out of order
    "is an enemy alien…Any person who",
    # a segment that is not in the source at all
    "No person shall be convicted…of any crime whatsoever",
])
def test_meaning_changing_or_fabricated_elision_is_rejected(quote):
    ok, _ = _elided_quote_ok(quote, SRC)
    assert not ok


def test_a_quote_with_no_ellipsis_is_not_handled_by_the_elision_path():
    assert _elided_quote_ok("No person shall be convicted", SRC) == (False, [])


# --------------------------------------------------------------------------
# Recodification mapping and in-force preference
# --------------------------------------------------------------------------

def test_full_correspondence_mapping_is_loaded(real_db):
    repealed = [p for p in real_db.all_provisions() if p.repealed_by]
    with_succ = sum(1 for p in repealed if real_db.successor(p.key))
    assert with_succ >= 1000, with_succ  # was 11 when only hand-tracked pairs were stored
    assert real_db.successor("CRPC§438") == ["BNSS§482"]
    assert real_db.is_mapping_verified("CRPC§438", "BNSS§482")


def test_split_section_is_annotated_with_every_successor_not_rewritten_to_one(real_db):
    from verify.verifier import CitationVerifier, Chunk
    assert sorted(real_db.successor("IPC§498A")) == ["BNS§85", "BNS§86"]
    ver = CitationVerifier(real_db, repair=True)
    out, rep = ver.verify("Cruelty is punishable under Section 498A of the IPC [1].",
                          [Chunk(1, "BNS§85", real_db.get("BNS§85").text, True)])
    assert not any(r.rule == "R2" for r in rep.repair_log.repairs)
    assert "BNS§85" in out and "BNS§86" in out


def test_in_force_preference_puts_the_successor_first(real_db):
    from retrieval.inforce import prefer_in_force
    ranked = [("IPC§302", 9.0), ("NIA§138", 5.0), ("BNS§103", 4.0)]
    assert [k for k, _ in prefer_in_force(ranked, real_db)] == ["BNS§103", "IPC§302", "NIA§138"]


def test_in_force_preference_only_reorders(real_db):
    from retrieval.inforce import prefer_in_force
    ranked = [("IPC§302", 9.0), ("NIA§138", 5.0)]  # successor not retrieved
    assert prefer_in_force(ranked, real_db) == ranked


def test_in_force_preference_improves_the_real_corpus_diagnostic(inforce):
    real = inforce["real"]
    assert (real["BM25 stemmed + in-force preference"]
            > real["BM25 stemmed"]), real
    assert real["annotated"] == real["n_pairs"]
