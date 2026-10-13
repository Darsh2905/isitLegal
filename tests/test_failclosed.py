"""
The fail-closed revision (2026-10): bare references, case citations, the
residual scan, lenticular markers, quotations split by the segmenter, and the
R2 enumeration corruption. Each mechanism is tested on the fault it exists to
catch AND the correct construct it must leave alone.
"""
import re

import pytest

from ingest.normalize import extract_citations, extract_bare_references
from ingest.case_citations import (CaseIndex, extract_case_names,
                                   extract_reporter_citations)
from verify.residual import residual_references
from verify.verifier import _segment, _MARKER


# ---- the probe set as a whole ----------------------------------------------

def test_every_fail_closed_fault_is_stripped(failclosed):
    bad = [r for r in failclosed["rows"] if r["expect"] == "strip" and not r["correct"]]
    assert not bad, bad
    assert failclosed["faults_n"] >= 22


def test_no_correct_citation_is_stripped(failclosed):
    bad = [r for r in failclosed["rows"] if r["expect"] == "keep" and not r["correct"]]
    assert not bad, bad
    assert failclosed["keep_n"] >= 24


# ---- residual scan -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The BNS 2023 replaced the IPC 1860.",
    "IPC vs BNS: the numbering changed.",
    "Part III of the Constitution guarantees fundamental rights.",
    "Within 24 hours of arrest the person must be produced.",
    "The fine may extend to Rs 5000 and 7 years.",
    "Under the Code of Criminal Procedure [5] it is compoundable.",
    "The Court's 3-judge bench decided it.",
    # false triggers found in live output and fixed
    "| Section(s) & title | Relevance |",
    "Post-1 July 2024 (BNS): death or life imprisonment.",
    "It is read with Exception 1 of the IPC.",
    "under Chapter XX of the Indian Penal Code (45 of 1860) except",
])
def test_residual_scan_ignores_non_references(text):
    assert residual_references(text, []) == []


@pytest.mark.parametrize("text", [
    "Section seven thousand of the IPC applies.",
    "The 7777th section of the IPC applies.",
    "In Ramesh v. State the Court held so.",
    "See (2019) 4 SCC 101.",
    "Clause (9) of Article 22 applies.",
])
def test_residual_scan_flags_reference_shaped_text(text):
    assert residual_references(text, [])


def test_residual_scan_respects_covered_spans():
    t = "Section 302 of the IPC punishes murder."
    spans = [(c.start, c.end) for c in extract_citations(t)]
    assert residual_references(t, spans) == []


# ---- parser additions -------------------------------------------------------

@pytest.mark.parametrize("text,key", [
    ("s 302 IPC", "IPC§302"), ("IPC's section 302", "IPC§302"),
    ("Section 302 (IPC)", "IPC§302"), ("IPC, s 302", "IPC§302"),
    ("Const §22", "CONST§22"),
    ("Clause (1) of Article 22", "CONST§22(1)"),
    ("sub-clause (a) of clause (1) of Article 19", "CONST§19(1)(a)"),
    ("sub-section (2) of section 41 CrPC", "CRPC§41(2)"),
])
def test_new_surface_forms_extract(text, key):
    assert key in [c.key for c in extract_citations(text)]


def test_bare_section_is_extracted_but_apostrophe_s_is_not():
    assert [b.section for b in extract_bare_references("Section 41 applies.")] == ["41"]
    assert extract_bare_references("The Court's 3 judges agreed.") == []


def test_act_first_symbol_list_keeps_every_member():
    keys = [c.key for c in extract_citations("the Bharatiya Nyaya Sanhita §318, §7777")]
    assert keys == ["BNS§318", "BNS§7777"]


# ---- markers and segmentation --------------------------------------------------

def test_lenticular_and_fullwidth_markers_are_markers():
    assert _MARKER.findall("a [1] b 【4】 c ［2］") == ["1", "4", "2"]


def test_segmenter_never_splits_inside_a_quotation():
    segs = [s for s, _ in _segment('It says "any records. information in it" [1]. Next.')]
    assert segs == ['It says "any records. information in it" [1].', "Next."]


# ---- case citations ---------------------------------------------------------------

class _J:
    def __init__(self, doc_id, name, cits):
        self.doc_id, self.case_name, self.citations = doc_id, name, cits


IDX = CaseIndex([
    _J("1", "Hitesh Verma vs The State Of Uttarakhand", ["AIRONLINE 2020 SC 812"]),
    _J("2", "Dr. Subhash Kashinath Mahajan vs The State Of Maharashtra",
       ["AIR 2018 SUPREME COURT 1498", "2018 (6) SCC 454"]),
    _J("3", "Navtej Singh Johar vs Union Of India Ministry Of Law And ...", []),
])


@pytest.mark.parametrize("text,doc", [
    ("In Hitesh Verma v. State of Uttarakhand the Court held", "1"),
    ("Navtej Singh Johar v. Union of India decided it", "3"),
    ("Fake Verma v. State of Uttarakhand held", None),
    ("Hitesh Verma v. State of Bihar held", None),
])
def test_case_names_resolve_only_when_both_sides_match(text, doc):
    names = extract_case_names(text)
    assert names and IDX.resolve_name(names[0]) == doc


def test_reporter_forms_normalise_to_the_same_key():
    a = extract_reporter_citations("(2018) 6 SCC 454")[0]
    b = extract_reporter_citations("2018 (6) SCC 454")[0]
    assert a.key == b.key and IDX.resolve_reporter(a) == "2"
    assert IDX.resolve_reporter(extract_reporter_citations("AIR 2018 SC 1498")[0]) == "2"
    assert IDX.resolve_reporter(extract_reporter_citations("(2019) 4 SCC 101")[0]) is None


def test_two_codes_compared_are_not_a_case_name():
    assert extract_case_names("IPC vs BNS: what changed") == []


def test_case_cited_in_a_retrieved_chunk_is_attested(real_db_fc):
    from verify.verifier import CitationVerifier, Chunk
    p = real_db_fc.get("ITA§66A")
    ver = CitationVerifier(real_db_fc)
    ans = ('Section 66A was "Held unconstitutional by the Supreme Court in '
           'Shreya Singhal v Union of India, AIR 2015 SC 1523" [1].')
    _, rep = ver.verify(ans, [Chunk(1, "ITA§66A", p.text, True)])
    assert rep.sentences_stripped == 0, [f.detail for f in rep.findings]
    _, rep = ver.verify("In Shreya Singhal v. Union of India the Court struck it down.", [])
    assert rep.sentences_stripped == 1


@pytest.fixture(scope="module")
def real_db_fc():
    from ingest.india_code import build_real_db
    return build_real_db()


# ---- repair R2 on enumerations ----------------------------------------------

def test_r2_does_not_corrupt_an_enumeration(real_db_fc):
    from verify.verifier import CitationVerifier, Chunk
    ver = CitationVerifier(real_db_fc, repair=True)
    out, rep = ver.verify("The accused was charged under sections 302 and 420 "
                          "of the IPC [1].",
                          [Chunk(1, "BNS§103", real_db_fc.get("BNS§103").text, True)])
    assert not re.search(r"\)[a-z]", out) and "(formerly" not in out
    assert rep.sentences_stripped == 0
    assert any("BNS§103" in a for a in rep.annotations)


def test_r2_rewrite_never_glues_the_next_word(real_db_fc):
    # P3 spans used to end in whitespace; R2 then emitted
    # "(formerly IPC Section 302 )punishes".
    from verify.verifier import CitationVerifier, Chunk
    ver = CitationVerifier(real_db_fc, repair=True)
    for text in ("IPC's section 302 punishes murder [1].",
                 "IPC Section 302 punishes murder [1]."):
        out, _ = ver.verify(text, [Chunk(1, "IPC§302", real_db_fc.get("IPC§302").text, True)])
        assert not re.search(r"\)[a-z]", out) and " )" not in out, out


def test_case_names_survive_segmentation():
    from verify.verifier import _segment
    segs = [s for s, _ in _segment("See K. P. D. v. State, 2007. Also Justice "
                                   "K.S. Puttaswamy (Retd.) v. Union of India held it.")]
    assert len(segs) == 2


def test_bare_reference_after_an_unknown_act_is_not_placed_elsewhere(real_db_fc):
    from verify.verifier import CitationVerifier
    ver = CitationVerifier(real_db_fc)
    _, rep = ver.verify("The power comes from the Indian Telegraph Act, 1885 "
                        "(especially §§ 5-7), the Information Technology Act, 2000.", [])
    assert rep.sentences_stripped == 1
