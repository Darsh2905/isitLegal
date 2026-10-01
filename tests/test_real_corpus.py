"""
Real-corpus ingestion (Tier 1). Tests the scraper, the correspondence-table
parser, and their assembly into MetadataDB -- not a claim that all 3,250
scraped provisions are individually legally verified (they are not; see
ingest/india_code.py's module docstring for what validation was actually
performed).

These tests are slower than the rest of the suite (a few seconds: HTML/PDF
parsing over ~2,700 provisions) and network-independent -- everything reads
from data/raw/, fetched once and cached, matching this project's
reproducibility rule (CLAUDE.md invariant 6: every number must be
reproducible by running a script in eval/, which here means from the cache,
not from a live site).
"""
import os
import re

import pytest

from ingest.scrape_indiankanoon import scrape_all, parse_akoma_html, numeric_gaps, RAW_DIR
from ingest.scrape_correspondence import parse_correspondence, build_verified_mapping
from ingest.india_code import build_real_db, build_sections_and_mapping


@pytest.fixture(scope="module")
def scraped():
    return scrape_all()


@pytest.fixture(scope="module")
def real_db():
    return build_real_db()


# --------------------------------------------------------------------------
# Scraper correctness -- spot checks against known real statutory text.
# These are the load-bearing tests: everything else assumes the scraper
# produces correct text, and that assumption is only as good as these checks.
# --------------------------------------------------------------------------

KNOWN_TEXT = {
    ("CONST", "21"): "No person shall be deprived of his life or personal "
                     "liberty except according to procedure established by law.",
    ("CONST", "14"): "The State shall not deny to any person equality before "
                     "the law or the equal protection of the laws within the "
                     "territory of India.",
}

# Article 334A (reservation of seats for women): a genuine markup variant
# found by a corpus-wide structural audit (Section~VIII), not a spot check
# -- all four of its subsections' real text lives in a SIBLING span the
# scraper never walked into, and the pre-fix output was the four bare
# subsection numerals concatenated: "1 2 3 4". Fixed by
# _resolve_orphaned_subsection. Exact-match, like CONST§21/§14 above, since
# the full text is short enough to pin precisely and this is the load-
# bearing case for that entire fix.
KNOWN_TEXT[("CONST", "334A")] = (
    "1 Notwithstanding anything in the foregoing provision of this Part or "
    "Part VIII, the provisions of the Constitution relating to the "
    "reservation of seats for women in the House of the People, the "
    "Legislative Assembly of a State and the Legislative Assembly of the "
    "National Capital Territory of Delhi shall come into effect after an "
    "exercise of delimitation is undertaken for this purpose after the "
    "relevant figures for the first census taken after commencement of the "
    "Constitution (One Hundred and Sixth Amendment) Act, 2023 have been "
    "published and shall cease to have effect on the expiration of a period "
    "of fifteen years from such commencement. 2 Subject to the provisions "
    "of articles 239AA, 330A and 332A, seats reserved for women in the "
    "House of the People, the Legislative Assembly of a State and the "
    "Legislative Assembly of the National Capital Territory of Delhi shall "
    "continue till such date as the Parliament may by law determine. 3 "
    "Rotation of seats reserved for women in the House of the People, the "
    "Legislative Assembly of a State and the Legislative Assembly of the "
    "National Capital Territory of Delhi shall take effect after each "
    "subsequent exercise of delimitation as the Parliament may by law "
    "determine. 4 Nothing in this article shall affect any representation "
    "in the House of the People, the Legislative Assembly of a State or the "
    "Legislative Assembly of the National Capital Territory of Delhi until "
    "the dissolution of the then existing House of the People, Legislative "
    "Assembly of a State or the Legislative Assembly of the National "
    "Capital Territory of Delhi."
)


@pytest.mark.parametrize("key", list(KNOWN_TEXT.keys()))
def test_verbatim_text_matches_known_law_exactly(scraped, key):
    act, num = key
    secs = {s.number: s for s in scraped[act]}
    assert num in secs, f"{act}§{num} missing from scrape"
    assert secs[num].text == KNOWN_TEXT[key]


def test_ipc_420_marginal_note_and_substance(scraped):
    s = {x.number: x for x in scraped["IPC"]}["420"]
    assert "Cheating" in s.marginal_note
    assert "deliver any property" in s.text
    assert "seven years" in s.text


def test_bns_103_punishment_for_murder(scraped):
    s = {x.number: x for x in scraped["BNS"]}["103"]
    assert s.marginal_note == "Punishment for murder"
    assert "death or imprisonment for life" in s.text


def test_no_editorial_commentary_leaks_into_body(scraped):
    # Class-A-style audit: Indian Kanoon's own added commentary ("Editorial
    # Comment", a "References" mini-list of external links) is not part of
    # the enacted text and must not appear in what V4 would check a quotation
    # against.
    for act, secs in scraped.items():
        for s in secs:
            assert "Editorial Comment" not in s.text, f"{act}§{s.number}"
            assert not s.text.rstrip().endswith("References"), f"{act}§{s.number}"


def test_no_html_markup_leaks_into_body(scraped):
    for act, secs in scraped.items():
        for s in secs:
            assert "<" not in s.text, f"{act}§{s.number}"
            assert "akn-" not in s.text, f"{act}§{s.number}"


def test_no_duplicated_body_content(scraped):
    # The Article 19 defect: nested content and sibling subsections both
    # captured the same clauses, producing the text followed by itself.
    tok = re.compile(r"[a-z0-9]+")
    for act, secs in scraped.items():
        for s in secs:
            toks = tok.findall(s.text.lower())
            n = len(toks)
            if n >= 20 and n % 2 == 0:
                assert toks[:n // 2] != toks[n // 2:], f"{act}§{s.number} duplicated"


def test_section_numbers_are_well_formed(scraped):
    pat = re.compile(r"^\d+[A-Za-z]*$")
    for act, secs in scraped.items():
        for s in secs:
            assert pat.match(s.number), f"{act}§{s.number!r} malformed"


def test_every_act_produced_a_plausible_section_count(scraped):
    # Loose bounds, not exact counts (the source can gain/lose a handful of
    # sections on re-scrape without that being a defect) -- but zero, or an
    # order of magnitude off, means the parser silently broke on that act's
    # markup, exactly the BNS defect found during development.
    bounds = {"IPC": (400, 700), "BNS": (300, 400), "CRPC": (400, 600),
             "BNSS": (400, 600), "IEA": (100, 250), "BSA": (100, 250),
             "NIA": (100, 200), "ITA": (80, 200), "ICA": (100, 250),
             "CONST": (350, 600)}
    for act, (lo, hi) in bounds.items():
        n = len(scraped.get(act, []))
        assert lo <= n <= hi, f"{act}: {n} sections, expected [{lo},{hi}]"


def test_bns_318_cheating_is_present_and_distinct_from_317(scraped):
    """
    A correction, recorded rather than quietly overwritten. This project first
    concluded BNS section 318 ("Cheating") was genuinely absent from Indian
    Kanoon's transcription -- 317 and 319 both present and correctly numbered
    either side of a real gap. That conclusion was WRONG, caught only later,
    while composing real generated answers for the generation experiment and
    noticing section 317's retrieved chunk ran on into text that was
    obviously section 318's cheating definition, not stolen-property text.

    The true defect: Indian Kanoon renders section 318 as TWO
    `<section class="akn-chapter"><h2>...` tags ("Chapter Of cheating", then
    "Chapter 318. Cheating.") instead of the usual akn-section/h3 pattern the
    other 3,250-odd sections use. akn-chapter was never in the parser's
    collected tag set, so section 318's heading was invisible and its body
    content -- real akn-subsection siblings -- was silently attributed to
    section 317, the last section the parser had actually opened. This is
    the SAME class of defect as the original BNS defect (a markup variant the
    parser didn't handle) but a more dangerous shape: that one produced zero
    sections and was caught by a section-count check; this one produced a
    plausible, correctly-numbered corpus with one section's content quietly
    merged onto its neighbour, which no structural sweep would catch without
    already suspecting something -- only reading real content against a real
    provision surfaced it. Confirmed to occur exactly once across all ten
    scraped Acts (see numeric_gaps and the module-level scan in
    ingest/scrape_indiankanoon.py's test coverage).
    """
    secs = {s.number: s for s in scraped["BNS"]}
    assert "317" in secs and "318" in secs and "319" in secs
    assert 318 not in numeric_gaps(scraped["BNS"])
    assert secs["317"].marginal_note == "Stolen property"
    assert secs["318"].marginal_note == "Cheating"
    assert "Chapter" not in secs["318"].text, "the mislabelled heading itself leaked into the body"
    assert "deceiving any person" in secs["318"].text
    assert "seven years" in secs["318"].text  # clause (4), the IPC§420-equivalent
    # 317 must no longer run on into 318's content.
    assert "deceiving any person" not in secs["317"].text
    assert len(secs["317"].text) < 2000


# --------------------------------------------------------------------------
# Correspondence table
# --------------------------------------------------------------------------

def test_correspondence_table_parses_a_substantial_number_of_rows():
    rows = parse_correspondence()
    assert len(rows) > 300


def test_correspondence_excludes_the_title_row_artifact():
    rows = parse_correspondence()
    assert not any(r.bns_section == "2023" and r.ipc_section == "1860" for r in rows)


def test_known_pairs_confirmed_where_the_source_actually_has_them():
    from data.seed_corpus import MAPPING
    updated = build_verified_mapping(MAPPING)
    by_pair = {(o, n): v for o, n, v in updated}
    for pair in [("IPC§302", "BNS§103"), ("IPC§420", "BNS§318"),
                ("IPC§375", "BNS§63"), ("IPC§376", "BNS§64")]:
        assert by_pair[pair] is True, f"{pair} should be confirmed"


def test_all_three_recodified_pairs_are_now_confirmed():
    # A matched series of three UP Police tables (BNS<->IPC, BNSS<->CRPC,
    # BSA<->IEA) now covers every pair this project tracks -- confirms the gap
    # that used to exist here (CRPC<->BNSS and IEA<->BSA had no source) is
    # closed, so it cannot silently reopen without this test catching it.
    from data.seed_corpus import MAPPING
    updated = build_verified_mapping(MAPPING)
    by_pair = {(o, n): v for o, n, v in updated}
    assert by_pair[("CRPC§41", "BNSS§35")] is True
    assert by_pair[("CRPC§154", "BNSS§173")] is True
    assert by_pair[("IEA§65B", "BSA§63")] is True


def test_mapping_never_silently_verifies_an_unconfirmed_pair():
    # A pair no table records must stay unverified -- build_verified_mapping
    # must not verify by coincidence of Act codes alone.
    updated = build_verified_mapping([("IPC§9999", "BNS§9999", False)])
    assert updated == [("IPC§9999", "BNS§9999", False)]


# --------------------------------------------------------------------------
# Assembly into MetadataDB
# --------------------------------------------------------------------------

def test_real_db_loads_and_is_large(real_db):
    assert real_db.count() > 2000


def test_real_db_every_provision_is_verbatim(real_db):
    assert all(p.verbatim for p in real_db.all_provisions())


def test_real_db_repeal_relationships_are_set(real_db):
    assert real_db.get("IPC§420").repealed_by == "BNS"
    assert real_db.get("BNS§103").repealed_by is None
    assert real_db.get("CRPC§154").repealed_by == "BNSS"


def test_real_db_successor_lookup_matches_verified_mapping(real_db):
    succ = real_db.successor("IPC§420")
    assert succ == ["BNS§318"]
    assert real_db.is_mapping_verified("IPC§420", "BNS§318") is True


def test_real_db_all_recodified_pairs_verified(real_db):
    assert real_db.is_mapping_verified("CRPC§41", "BNSS§35") is True
    assert real_db.is_mapping_verified("IEA§65B", "BSA§63") is True


def test_real_db_is_independent_of_the_toy_corpus():
    # The two corpora must not interfere with each other -- building one must
    # not mutate or invalidate the other (both use in-memory sqlite).
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.metadata_db import build_seed_db
    toy = build_seed_db()
    real = build_real_db()
    assert toy.count() == 35
    assert real.count() > 2000
    assert toy.get("IPC§420").verbatim is False   # toy corpus: paraphrase
    assert real.get("IPC§420").verbatim is True   # real corpus: verbatim


def test_bnss_is_included_and_plausibly_sized(real_db):
    # BNSS was initially missed (early searches surfaced only individual-
    # section URLs) and closed once the full-act page was found. This
    # confirms it stayed closed: a plausible section count with the known
    # landmark sections present and correctly numbered.
    bnss = [p for p in real_db.all_provisions() if p.act == "BNSS"]
    assert 400 <= len(bnss) <= 600
    assert real_db.get("BNSS§35").marginal_note == "When police may arrest without warrant"
    assert real_db.get("BNSS§173") is not None


# --------------------------------------------------------------------------
# All three correspondence tables (generalised parser)
# --------------------------------------------------------------------------

def test_all_three_correspondence_tables_parse():
    from ingest.scrape_correspondence import parse_all, TABLE_SOURCES
    all_rows = parse_all()
    assert len(all_rows) == 3
    for new_act, old_act, _fname, _url in TABLE_SOURCES:
        assert (new_act, old_act) in all_rows
        assert len(all_rows[(new_act, old_act)]) > 100


def test_bnss_crpc_table_confirms_known_pairs():
    from ingest.scrape_correspondence import parse_all
    rows = parse_all()[("BNSS", "CRPC")]
    pairs = {(r.old_section, r.new_section) for r in rows}
    assert ("41", "35") in pairs
    assert ("154", "173") in pairs


def test_bsa_iea_table_confirms_known_pair():
    from ingest.scrape_correspondence import parse_all
    rows = parse_all()[("BSA", "IEA")]
    pairs = {(r.old_section, r.new_section) for r in rows}
    assert ("65B", "63") in pairs


def test_bnss_body_text_matches_known_law(scraped):
    s = {x.number: x for x in scraped["BNSS"]}["35"]
    assert s.marginal_note == "When police may arrest without warrant"
    assert "without an order from a Magistrate" in s.text


def test_real_db_mapping_verification_rate(real_db):
    # All 11 tracked pairs are now confirmed. IPC§379->BNS§303 used to be
    # excluded: the source table specifies the more precise BNS§303(2), and
    # an exact-tuple match against our own section-level "BNS§303" entry
    # failed. ingest.india_code._pair_confirmed fixed this -- IPC§379's
    # correspondence genuinely is confirmed by the source, just at finer
    # granularity than this corpus indexes (one chunk per section, not per
    # sub-section: BNS§303's single chunk already contains both the
    # definition and punishment sub-sections IPC split across 378/379). See
    # _pair_confirmed's docstring for the full reasoning and the guard
    # against the reverse case (a bare source entry must never confirm a
    # sub-section-qualified claim we didn't make).
    from data.seed_corpus import MAPPING
    verified = sum(1 for old, new, _ in MAPPING
                  if real_db.is_mapping_verified(old, new))
    assert verified == 11


# --------------------------------------------------------------------------
# Third adversarial audit (see tests/test_evasion.py for the milder half of
# this same finding). Found live, inspecting a genuinely generated app.py
# Tab 6 answer during the demo build: it wrote "the Indian Evidence Act
# §113‑B" (a non-breaking hyphen, U+2011) and separately this project's own
# worked example, "Section 65-B" (a plain ASCII hyphen), is the single most
# common real-world citation style for the Evidence Act's electronic-records
# provision. Neither is reproducible against the 35-provision toy corpus,
# which has no same-numeral collision pair -- the real 3,263-provision
# corpus does, and the danger is worse than a blind spot: instead of the
# fabricated/mistyped reference going unchecked, it silently validated
# against a real but wholly UNRELATED shorter provision that happened to
# share the leading digits.
# --------------------------------------------------------------------------

def test_no_hyphenated_citation_key_in_the_real_corpus(real_db):
    # The premise the fix depends on: every compound-suffix key in the real
    # corpus joins the letter directly ("IEA§65B"), never with a hyphen
    # ("IEA§65-B"). If this ever stops being true, _clean_section's decision
    # to strip the hyphen rather than keep it needs revisiting.
    assert not any("-" in p.key for p in real_db.all_provisions())


def test_ascii_hyphenated_suffix_resolves_to_the_correct_provision(real_db):
    from ingest.normalize import extract_citations
    cits = extract_citations(
        "Electronic records are admissible under Section 65-B of the "
        "Indian Evidence Act.")
    assert len(cits) == 1
    assert cits[0].key == "IEA§65B"
    hit = real_db.get(cits[0].key)
    assert hit is not None
    assert hit.marginal_note == "Admissibility of electronic records"
    # The wrong provision this used to silently resolve to instead: P4's
    # number token had no hyphen branch at all (even for a plain ASCII
    # hyphen), so it truncated to "65", which is itself a real, different
    # provision -- confirmed distinct here so a regression that reintroduces
    # the truncation would fail this assertion rather than pass by accident.
    collision = real_db.get("IEA§65")
    assert collision is not None
    assert collision.marginal_note != hit.marginal_note


def test_non_breaking_hyphen_suffix_resolves_correctly_not_to_a_collision(real_db):
    # The exact construct a live Tab 6 answer emitted: U+2011 (non-breaking
    # hyphen), not the ASCII "-" the test above uses. Confirms the fix is a
    # genuine Unicode-class normalisation, not a single-character coincidence.
    from ingest.normalize import extract_citations
    cits = extract_citations(
        "the Indian Evidence Act §113‑B")
    assert len(cits) == 1
    assert cits[0].key == "IEA§113B"
    # The point of the test: never resolve to the unrelated IEA§113. This
    # test previously ALSO asserted IEA§113B was absent from the corpus --
    # that recorded a scraper defect, not the law: s.113-B (presumption as
    # to dowry death) is real, and was missing only because its hyphenated
    # run-in heading ("113-B.") was never split out of s.113
    # (scrape_indiankanoon._RUN_IN).
    found = real_db.get(cits[0].key)
    assert found is not None
    assert found.marginal_note == "Presumption as to dowry death"
    collision = real_db.get("IEA§113")
    assert collision is not None
    assert collision.marginal_note == "Proof of cession of territory"


def test_act_first_enumeration_with_repeated_symbol_extracts_every_member(real_db):
    # "the Bharatiya Nyaya Sanhita §80, §118" -- BNS§118 vanished entirely
    # (not partially) because the list-continuation grammar couldn't consume
    # a literal "§" where the next atom needed to start with a digit, so the
    # whole repetition group matched zero times and P4B silently fell back
    # to the list's first member only.
    from ingest.normalize import extract_citations
    cits = extract_citations(
        "This is contained in the Bharatiya Nyaya Sanhita §80, §118.")
    keys = {c.key for c in cits}
    assert keys == {"BNS§80", "BNS§118"}
    for k in keys:
        assert real_db.get(k) is not None


def test_act_name_then_parenthesised_section_resolves_on_real_text(real_db):
    # Found live: Groq's real, independently-generated answer to "Is
    # defamation a criminal offence in India?" wrote "the Indian Penal Code
    # (Section 500)" -- extracted zero citations before the fix, on a real
    # provision (IPC§500 exists in the corpus, superseded by BNS with no
    # recorded successor for this specific section).
    from ingest.normalize import extract_citations
    cits = extract_citations(
        "It is punishable under the Indian Penal Code (Section 500).")
    assert len(cits) == 1
    assert cits[0].key == "IPC§500"
    assert real_db.get("IPC§500") is not None


# --------------------------------------------------------------------------
# Corpus-wide structural audit (Section~VIII): unlike every defect above,
# found not by adversarial construction or by inspecting generated text, but
# by sweeping every one of the 3,263 real provisions for structural
# anomalies -- length outliers, self-duplication, truncation -- the kind of
# thing CLAUDE.md's own roadmap names as the highest-value unaudited surface
# in the project. Two distinct scraper defects came out of it, affecting a
# confirmed 12 provisions across four Acts.
# --------------------------------------------------------------------------

def test_bare_numeral_subsection_intro_is_rescued_from_its_sibling(real_db):
    # BNS§326 ("Mischief by injury..."): the intro clause's own nested body
    # was "Whoever commits mischief by," with clauses (a)/(b)/(c) living in
    # a sibling span _resolve_orphaned_subsection did not originally check
    # for (only a BARE NUMERAL triggered it) -- broadened to also catch a
    # list-introducing clause that ends at the comma/dash.
    p = real_db.get("BNS§326")
    assert p is not None
    assert p.text.strip() != "Whoever commits mischief by,"
    assert "diminution of the supply of water" in p.text  # clause (a)
    assert len(p.text) > 2000


def test_nested_subsections_are_not_double_counted(real_db):
    # IEA§146 ("Questions lawful in cross-examination"): the section-level
    # grab already includes every nested subsection's text (the ordinary,
    # correctly-working case for this markup variant), and the parser's
    # main loop THEN separately walks each subsection tag again and
    # re-appends it -- producing an exact duplicate _dedup_halves's
    # original exact-half check could not see, because a unique intro
    # clause ("...which tend -") comes before the duplicated part, making
    # the two "halves" unequal in length.
    p = real_db.get("IEA§146")
    assert p is not None
    assert p.text.count("to test his veracity") == 1
    assert p.text.count("to shake his credit") == 1


def test_orphaned_subsection_rescue_does_not_reintroduce_duplication(real_db):
    # BNSS§266 ("Evidence for defence"): the mirror-image interaction
    # between the two fixes above -- subsection (3)'s own nested body
    # resolved to a near-empty stub, EVEN THOUGH the section-level grab
    # already had its full text (the earlier IEA§146 case did not).
    # Rescuing (3) from its sibling without also recognising that the
    # section-level grab already contains subsections (1) and (2) would
    # have reintroduced exactly the duplication the fix above removes --
    # this is the case _dedup_parts exists for, operating on the parser's
    # own body-parts list rather than the joined string, where the
    # ambiguity does not arise.
    p = real_db.get("BNSS§266")
    assert p is not None
    assert p.text.count("accused shall then be called") == 1
    assert p.text.count("reasonable expenses incurred by the witness") == 1
    assert "Magistrate may, before summoning any witness" in p.text


def test_corpus_wide_audit_heuristic_false_positives_are_real_text(real_db):
    # The other side of the same audit, reported honestly rather than
    # silently dropped: three provisions flagged by the same "repeated
    # opening text" heuristic that found the two real bugs above turned out
    # to be genuine statutory drafting, not scraper defects -- IPC§99 opens
    # its first two clauses with the identical formula, verbatim, for two
    # different actors (a public servant, then someone acting under a
    # public servant's direction), before a third, differently-worded
    # clause and two Explanations. This corpus preserves that real
    # repetition rather than being "fixed" to remove it.
    p = real_db.get("IPC§99")
    assert p is not None
    assert p.text.count(
        "There is no right of private defence against an act which does "
        "not reasonably cause the apprehension of death") == 2
    assert "no right of private defence in cases in which there is time" in p.text
    assert "Explanation 2" in p.text


# --------------------------------------------------------------------------
# Session N+1 fixes: separator-glued clause text, heading-embedded amendment
# history, silent whole-section loss, and non-current text (prior-version /
# state-amendment tables) leaking into a section's operative body. Found by
# the corpus-wide structural audit described in CLAUDE.md Section 6.
# --------------------------------------------------------------------------

def test_adjacent_clause_spans_are_not_glued_together(real_db):
    # get_text()'s default separator inserts NOTHING between adjacent
    # sibling spans; the source markup for enumerated list clauses
    # routinely carries no whitespace of its own between them either, so
    # "...or</span><span>is of such a nature..." was rendered verbatim as
    # "orIS OF such a nature" -- corrupting exactly the text this
    # project's own quotation checker (V4) exists to verify. Fixed by
    # passing separator=" " to get_text() in _body_text.
    p = real_db.get("ICA§23")
    assert p is not None
    assert "oris of such a nature" not in p.text
    assert "orinvolves" not in p.text
    assert "orthe Court regards" not in p.text
    assert "or is of such a nature that, if permitted" in p.text
    assert "or involves or implies, injury" in p.text


def test_heading_amendment_history_does_not_pollute_the_marginal_note(real_db):
    # CRPC§166A's marginal note carried its own amendment history
    # ("[Inserted by Act 10 of 1990, ...]") because _heading_text (unlike
    # _body_text) called h.get_text() directly, bypassing the same
    # editorial-remark stripping. CRPC§330 is the more severe sibling
    # case: the ENTIRE pre-amendment text of the section, unmarked by any
    # data-status="editorial" span, dumped directly into the heading.
    for act, num, forbidden in [
        ("CRPC", "166A", "Inserted by Act"),
        ("CRPC", "330", "Prior to its substitution"),
        ("CRPC", "478", "Substituted by"),
    ]:
        p = real_db.get(f"{act}§{num}")
        assert p is not None, f"{act}§{num} missing"
        assert forbidden not in p.marginal_note, (
            f"{act}§{num} marginal note still carries amendment-history "
            f"pollution: {p.marginal_note!r}")
        assert not p.marginal_note.startswith("["), (
            f"{act}§{num} marginal note has a dangling leading bracket: "
            f"{p.marginal_note!r}")


def test_sections_whose_rule_lives_entirely_in_the_heading_are_not_lost(real_db):
    # ICA§184's <section> tag has nothing outside its <h3> at all --
    # _body_text correctly excludes heading text from body (that's right
    # for every ordinary section), which here left body empty and
    # flush()'s `if body and ...` check silently dropped the entire
    # section: not a wrong citation, an invisible one. ICA§68 is the same
    # bug in milder disguise (a separate Illustrations block makes body
    # non-empty, so the section survives but silently missing its actual
    # rule). IPC§56/58/161/165/492 are genuinely repealed sections with no
    # operative rule left, but a repeal notice deserves to exist as a
    # citable, honest record -- not vanish with no trace.
    agent = real_db.get("ICA§184")
    assert agent is not None, "ICA§184 must not be silently dropped"
    assert agent.marginal_note == "Who may be an agent"
    assert "no person who is not of the age of majority" in agent.text

    necessaries = real_db.get("ICA§68")
    assert necessaries is not None
    assert "If a person, incapable of entering into a contract" in necessaries.text
    assert "entitled to be reimbursed from the property" in necessaries.text
    # The rescued operative rule must come BEFORE the section's own
    # Illustrations block, not after it -- rule first, examples second.
    assert necessaries.text.index("entitled to be reimbursed from the property"
                                   ) < necessaries.text.index("Illustrations")

    for num in ["56", "58", "161", "165", "492"]:
        p = real_db.get(f"IPC§{num}")
        assert p is not None, f"IPC§{num} (repealed) must not be silently dropped"
        assert "Rep. by" in p.text, f"IPC§{num} should record its repeal notice"


def test_long_marginal_note_threshold_does_not_touch_legitimate_titles(real_db):
    # Six sections in this corpus legitimately use an em dash inside a
    # SHORT marginal note (IPC§70, 115, 116, 153; NIA§16; ICA§23); the
    # heading-rescue mechanism above must not mistake any of them for the
    # heading-embedded-rule pattern it exists to catch. All stay well
    # under the 150-char threshold that separates them from ICA§68 (381).
    for act, num, note_contains in [
        ("IPC", "70", "Fine leviable within six years"),
        ("IPC", "115", "Abetment of offence punishable with death"),
        ("IPC", "116", "Abetment of offence punishable with imprisonment"),
        ("IPC", "153", "Wantonly giving provocation"),
        ("NIA", "16", "Indorsement"),
        ("ICA", "23", "What consideration and objects are lawful"),
    ]:
        p = real_db.get(f"{act}§{num}")
        assert p is not None
        assert len(p.marginal_note) <= 150
        assert note_contains in p.marginal_note


def test_orphaned_closing_bracket_stripped_only_when_unbalanced(real_db):
    # _match_leading_number's leading-"[" strip left a dangling, unmatched
    # "]" behind on five repealed-section notes (IPC§56/58/161/165/492:
    # "...for life.] Rep. by..."). Fixed by dropping that first "]" only
    # when the note has no surviving "[" at all -- a note where both
    # brackets survive (IEA§35's "public [record of an electronic
    # record]", CONST's "... [Repealed]") is a real, balanced bracket the
    # drafting itself uses and must be left alone.
    for num in ["56", "58", "161", "165", "492"]:
        p = real_db.get(f"IPC§{num}")
        assert p is not None
        assert "]" not in p.marginal_note, f"IPC§{num} note: {p.marginal_note!r}"

    balanced = real_db.get("IEA§35")
    assert balanced is not None
    assert "[record of an electronic record]" in balanced.marginal_note
    repealed_article = real_db.get("CONST§31D")
    assert repealed_article is not None
    assert "[Repealed]" in repealed_article.marginal_note


def test_prior_version_table_text_does_not_leak_into_current_body(real_db):
    # Indian Kanoon renders a superseded sub-section's OLD wording as a
    # plain akn--table-container table (IEA§119, IEA§114A) or, in one
    # case, inline bracketed prose (CRPC§127) sitting right inside the
    # CURRENT section -- not marked data-status="editorial". Left in, a
    # fabricated quote drawn from the superseded wording would incorrectly
    # validate against a citation to the current section: exactly the
    # failure V4 exists to prevent. Only the unambiguous "prior to (its)
    # substitution ... read as under" marker is stripped; genuinely
    # current tables (the CRPC§320 compounding-of-offences table) are
    # deliberately left untouched -- see _body_text's docstring for what
    # remains a stated, unfixed gap (state/UT amendment tables).
    for act, num in [("IEA", "119"), ("IEA", "114A"), ("CRPC", "127"),
                      ("CRPC", "54"), ("CRPC", "309")]:
        p = real_db.get(f"{act}§{num}")
        assert p is not None
        assert "prior to its substitution" not in p.text.lower()
        assert "read as under" not in p.text.lower()

    compounding = real_db.get("CRPC§320")
    assert compounding is not None
    assert "Person by whom offence may be compounded" in compounding.text


def test_soft_hyphen_does_not_split_words(real_db):
    # U+00AD SOFT HYPHEN is an invisible print-only hyphenation hint the
    # source left embedded mid-word (IPC§31: "docu\xadment"). Invisible to
    # a human reader, but a real character that silently breaks exact-
    # token quotation matching (V4) against a claim quoting the word
    # normally. Found across 137 occurrences in the ten scraped Acts;
    # deleted outright in _clean (unlike \xa0, a soft hyphen has no
    # visible-character equivalent to replace it with).
    p = real_db.get("IPC§31")
    assert p is not None
    assert "\xad" not in p.text
    assert "\xad" not in p.marginal_note
    assert p.text == 'The words “a will” denote any testamentary document.'


def test_no_soft_hyphen_anywhere_in_the_real_corpus(scraped):
    for act, secs in scraped.items():
        for s in secs:
            assert "\xad" not in s.text, f"{act}§{s.number} still has a soft hyphen in text"
            assert "\xad" not in s.marginal_note, f"{act}§{s.number} still has a soft hyphen in note"


def test_rescued_sections_do_not_change_the_expected_gap_pattern(scraped):
    # ICA§184 and the five IPC sections rescued this round were previously
    # silent, untracked gaps in the numeric sequence (numeric_gaps only
    # sees section numbers that are ACTUALLY missing, so their absence
    # never showed up there at all -- this is a belt-and-suspenders check
    # that they are genuinely present now, not a claim about gaps).
    ica_by_num = {s.number: s for s in scraped["ICA"]}
    ipc_by_num = {s.number: s for s in scraped["IPC"]}
    assert "184" in ica_by_num
    assert "68" in ica_by_num
    for num in ["56", "58", "161", "165", "492"]:
        assert num in ipc_by_num
