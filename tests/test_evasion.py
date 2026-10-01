"""
Parser-evasion resistance (paper VII-G).

Class-A bugs: a check is bypassed because the parser never sees the citation.
Each probe conceals a fault inside a construct that previously defeated the
parser (enumerations, abbreviation-induced sentence splits, unterminated
quotations).
"""


def test_every_concealed_fault_is_caught(evasion):
    assert evasion["fn"] == 0
    assert evasion["tp"] == 21
    assert evasion["recall"] == 1.0


def test_no_false_positives_on_valid_constructs(evasion):
    assert evasion["fp"] == 0
    assert evasion["tn"] == 8
    assert evasion["precision"] == 1.0


def test_enumerated_references_extract_every_citation(evasion):
    # The most serious Class-A bug: "sections 302 and 304 of the IPC" once
    # extracted ZERO citations, making a fabricated section in a list wholly
    # invisible to V2. Asserting the COUNT, not just the verdict, is what
    # detects a regression to partial extraction.
    by_pid = {r["pid"]: r for r in evasion["rows"]}
    assert by_pid["ev-list-1"]["citations_found"] >= 2
    assert by_pid["ev-list-2"]["citations_found"] >= 2
    assert by_pid["ev-list-3"]["citations_found"] >= 3


def test_citations_survive_abbreviation_splits(evasion):
    by_pid = {r["pid"]: r for r in evasion["rows"]}
    assert by_pid["ev-abbrev-1"]["citations_found"] >= 1
    assert by_pid["ev-abbrev-2"]["citations_found"] >= 1


def test_unterminated_quotation_does_not_bypass_v4(evasion):
    assert "unsupported_quote" in evasion["caught"]["ev-quote-1"]


# --------------------------------------------------------------------------
# Second adversarial audit. Each construct below concealed a fabricated
# section from the existence check entirely -- the parser extracted nothing,
# so V2 never ran and no error was raised. Asserting the extracted COUNT is
# what detects a regression to partial extraction, which a verdict-only
# assertion would miss.
# --------------------------------------------------------------------------

def _row(evasion, pid):
    return {r["pid"]: r for r in evasion["rows"]}[pid]


def test_act_first_enumerations_extract_every_member(evasion):
    # "BNS sections 318 and 7777" -- the mirror of the enumeration defect.
    for pid in ("ev-actlist-1", "ev-actlist-2"):
        assert _row(evasion, pid)["citations_found"] >= 2, pid


def test_alternative_separators_extract_every_member(evasion):
    # Slash, semicolon and r/w are standard in Indian legal drafting and were
    # all absent from the separator set.
    for pid in ("ev-slash-1", "ev-slash-2", "ev-semi-1", "ev-rw-1"):
        assert _row(evasion, pid)["citations_found"] >= 2, pid
        assert "nonexistent_provision" in evasion["caught"][pid], pid


def test_markup_does_not_conceal_a_citation(evasion):
    # Generators emit markdown emphasis constantly; a fabricated section
    # inside it was invisible.
    for pid in ("ev-md-1", "ev-md-2", "ev-tick-1", "ev-paren-1", "ev-fn-1"):
        assert _row(evasion, pid)["citations_found"] >= 1, pid
        assert "nonexistent_provision" in evasion["caught"][pid], pid


def test_invisible_characters_do_not_conceal_a_citation(evasion):
    # A zero-width space between the number and the act name defeated every
    # pattern. This is a deliberate-evasion vector, not a natural one.
    assert _row(evasion, "ev-zwsp-1")["citations_found"] >= 1
    assert "nonexistent_provision" in evasion["caught"]["ev-zwsp-1"]


def test_controls_for_the_new_constructs_are_not_flagged(evasion):
    # The fixes widened the parser; these confirm it did not become trigger-
    # happy on the same syntax carrying only real, in-force provisions.
    for pid in ("ev-ok-3", "ev-ok-4", "ev-ok-5", "ev-ok-6"):
        assert _row(evasion, pid)["flagged"] is False, pid


# --------------------------------------------------------------------------
# Third adversarial audit. Found live, inspecting a genuinely generated
# app.py Tab 6 answer during the demo build -- not by extending these probes
# adversarially, the way the first two rounds were found. See
# tests/test_real_corpus.py for the more severe half of this same finding: a
# silent false VALIDATION against the wrong real provision, reproducible
# only against the real corpus, which this toy-corpus harness cannot show.
# --------------------------------------------------------------------------

def test_repeated_section_symbol_does_not_drop_a_list_member(evasion):
    # "the Bharatiya Nyaya Sanhita §318, §7777" -- a generator that repeats
    # the section symbol per list item rather than dropping it after the
    # first defeated the act-first bare-enumeration pattern entirely: the
    # continuation group couldn't consume a literal "§" where it expected a
    # digit, so it matched zero repetitions and the whole list collapsed to
    # its first member. The fabricated second member was invisible, not
    # partially reported.
    row = _row(evasion, "ev-actlist-repeat-1")
    assert row["citations_found"] == 2
    assert "nonexistent_provision" in evasion["caught"]["ev-actlist-repeat-1"]


def test_hyphenated_compound_suffix_is_not_wrongly_flagged(evasion):
    # "Section 43-A" is a real, in-force, correctly-cited provision. Before
    # the fix, the canonicalisation step kept the hyphen ("ITA§43-A")
    # instead of dropping it to match how every key in the corpus is
    # actually stored ("ITA§43A"), so a real citation in an extremely
    # common real-world citation style would have been flagged as
    # fabricated.
    assert _row(evasion, "ev-hyphensuffix-ok-1")["flagged"] is False
    assert _row(evasion, "ev-hyphensuffix-ok-1")["citations_found"] == 1


# --------------------------------------------------------------------------
# Fourth adversarial audit. Found live from the actual live-generation
# experiment (eval/run_live_generation_eval.py) -- the second construct in
# this project found through real usage rather than deliberate probe
# construction. Groq's real answer to a defamation query wrote "the Indian
# Penal Code (Section 500)" and it extracted zero citations.
# --------------------------------------------------------------------------

def test_act_name_then_parenthesised_section_extracts_the_citation(evasion):
    # P3 (act name, then a section reference) required a comma or "of the"
    # between the act and the section word; a literal "(" satisfied
    # neither, so the whole clause extracted nothing -- not a partial match.
    assert _row(evasion, "ev-actparen-1")["citations_found"] == 1
    assert "nonexistent_provision" in evasion["caught"]["ev-actparen-1"]


def test_act_name_then_parenthesised_section_is_not_wrongly_flagged(evasion):
    assert _row(evasion, "ev-actparen-ok-1")["flagged"] is False
    assert _row(evasion, "ev-actparen-ok-1")["citations_found"] == 1
