"""
Assembles the real statutory corpus (Tier 1) from the scraped sources into the
same schema `data/seed_corpus.py` uses, so it loads through the existing
`MetadataDB` unchanged.

RELATIONSHIP TO data/seed_corpus.py. This module does NOT replace the seed
corpus, despite CLAUDE.md's original roadmap wording ("replace
data/seed_corpus.py"). The seed corpus is kept, unchanged, because it serves a
distinct and still-necessary purpose this module cannot: a small, hand-authored
set of provisions built specifically so that PLANTED FAULTS (a fabricated
section, a superseded reference, a meaning-inverting quotation) can be placed
precisely where the probe sets expect them. Every existing verifier/repair/
support probe (eval/verifier_probes.py) is keyed to that corpus's exact
content, and rebuilding it against real statutory text would not make those
tests more meaningful -- it would just make them fragile against a scrape that
can change. The two corpora now serve different, complementary roles:

  data/seed_corpus.py  -- synthetic, small, fault-injectable. Answers "does
                          each check fire on the fault it is specified to
                          catch?"
  ingest/india_code.py -- real, large, government-enacted text. Answers "what
                          does retrieval/quotation/cross-reference behaviour
                          look like away from a toy-sized, hand-curated
                          collection?" -- the question the 35-provision corpus
                          could not answer no matter how it was measured.

SOURCE AND VALIDATION. Provisions come from ingest/scrape_indiankanoon.py
(see that module for why Indian Kanoon rather than India Code, and the parser
defects found and fixed while building it -- a missing markup variant that
silently produced zero BNS sections, a nested-clause bug that silently
returned the wrong body text for one IT Act section, and a duplicated-content
bug on heavily-amended Constitution articles). Every provision here is
`verbatim=True`. All ten target Acts (IPC, BNS, CRPC, BNSS, IEA, BSA, NIA, ITA,
ICA, Constitution) are included -- BNSS's full-act page was initially missed
because early searches surfaced only individual-section URLs; the full act
turned out to exist on Indian Kanoon in the same Akoma Ntoso structure the
other nine Acts use (indiankanoon.org/doc/91117739/), so it reuses the same
tested parser rather than needing a bespoke one for the harder two-column PDF
bill format its official source publishes.

Validation performed: automated structural sweep over all scraped sections (no
HTML/markup leakage, no editorial-commentary leakage, no body under 15
characters beyond two known thin residuals, no malformed section-number field,
zero unexplained numeric gaps in BNSS) plus targeted spot-checks of specific
known provisions (IPC 420, BNS 103/319, Articles 21/14/19, NIA 138, IEA 65,
BSA 63, ITA 66, BNSS 35/173) against their real statutory text. This is NOT an
exhaustive legal review of every provision -- report results accordingly,
exactly as the seed corpus's own README caveat already does for its much
smaller, paraphrased set.

RECODIFICATION MAPPING. All three recodified pairs -- BNS<->IPC, BNSS<->CRPC,
BSA<->IEA -- are cross-checked against real, citable government sources (a
matched series of three UP Police "Corresponding Section Table" documents; see
ingest/scrape_correspondence.py) and marked verified=True only where the
matching table confirms the exact pair.
"""

import re
import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.scrape_indiankanoon import scrape_all, ACT_SOURCES
from ingest.scrape_correspondence import (parse_all as parse_all_correspondence,
                                          TABLE_SOURCES as CORRESPONDENCE_SOURCES)

# Acts superseded by the 2023 recodification, and by what.
_REPEALED_BY = {"IPC": "BNS", "CRPC": "BNSS", "IEA": "BSA"}

# The known cross-Act section-number pairs this project already tracked before
# real sources existed to check them (data/seed_corpus.py MAPPING). Kept
# duplicated here, deliberately, rather than imported: this module's MAPPING
# is built from the real correspondence tables directly, and importing the
# seed corpus's hand list would blur which entries came from which source.
# Format: (new_act, new_num, old_act, old_num).
_KNOWN_PAIRS = [
    ("BNS", "103", "IPC", "302"), ("BNS", "318", "IPC", "420"),
    ("BNS", "303", "IPC", "378"), ("BNS", "303", "IPC", "379"),
    ("BNS", "63", "IPC", "375"), ("BNS", "64", "IPC", "376"),
    ("BNS", "80", "IPC", "304B"), ("BNS", "356", "IPC", "499"),
    ("BNSS", "35", "CRPC", "41"), ("BNSS", "173", "CRPC", "154"),
    ("BSA", "63", "IEA", "65B"),
]


def _pair_confirmed(new_act, new_num, old_act, old_num, confirmed) -> bool:
    """
    Whether a _KNOWN_PAIRS entry is confirmed by the real correspondence
    table -- exact match, OR the table's entry is confirmed at a FINER
    granularity than this corpus indexes.

    Concretely: IPC's theft offence was two separate sections (378 the
    definition, 379 the punishment); BNS consolidates both into one section,
    303, with the definition and punishment as its two sub-sections. The
    official correspondence table is precise about this -- it confirms
    378->303 (bare) but 379->303(2) (with the sub-section). This project's
    corpus, however, scrapes one chunk per SECTION, not per sub-section
    (Section IV-A "Chunking"), so there is no "BNS§303(2)" provision to
    point a mapping at -- BNS§303's single chunk already contains both
    sub-sections' text. Recording the mapping at section granularity
    (IPC§379 -> BNS§303, not BNS§303(2)) is therefore the CORRECT choice
    for this corpus's own indexing, not an imprecision to fix by inventing
    a sub-section key nothing else in the corpus has. What was actually
    wrong was the verification check: an exact-tuple match against
    "303(2)" never matches our own recorded "303", so a real, source-backed
    confirmation was being reported as unverified. This function fixes
    that by accepting the source's sub-section-qualified entry as
    confirming our section-level one, WHEN our own recorded mapping omits
    a sub-section (never the reverse -- a bare source entry does not
    confirm a sub-section-qualified claim we didn't make).
    """
    if (new_act, new_num, old_act, old_num) in confirmed:
        return True
    if "(" in new_num:
        return False  # we claimed sub-section precision; only an exact match counts
    prefix = new_num + "("
    return any(c_act == new_act and c_old_act == old_act and c_old == old_num
              and c_new.startswith(prefix)
              for c_act, c_new, c_old_act, c_old in confirmed)


def build_sections_and_mapping() -> Tuple[list, list, dict]:
    """
    Returns (SECTIONS, MAPPING, STATUS_NOTES) in exactly the shape
    `MetadataDB.load_seed` / `data/seed_corpus.py` already use.
    """
    scraped = scrape_all()

    sections: List[tuple] = []
    for name, fname, act, url, word in ACT_SOURCES:
        for s in scraped.get(act, []):
            repealed_by = _REPEALED_BY.get(act)
            sections.append((act, s.number, s.marginal_note, s.text,
                             repealed_by, True))  # verbatim=True: real text

    # --- recodification mapping, verified against the real correspondence tables ---
    all_rows = parse_all_correspondence()
    confirmed = set()
    for (new_act, old_act), rows in all_rows.items():
        for r in rows:
            confirmed.add((new_act, r.new_section, old_act, r.old_section))

    mapping: List[Tuple[str, str, bool]] = []
    for new_act, new_num, old_act, old_num in _KNOWN_PAIRS:
        verified = _pair_confirmed(new_act, new_num, old_act, old_num, confirmed)
        mapping.append((f"{old_act}§{old_num}", f"{new_act}§{new_num}", verified))

    # The FULL correspondence tables, not only the 11 hand-tracked pairs above.
    # Every row was already parsed (it populates `confirmed`) and then thrown
    # away: 1,241 of the corpus's 1,252 repealed provisions had no recorded
    # successor, so R2 could not repair and V3 could only say "no mapping
    # recorded" for all but eleven sections. Each row comes straight from the
    # official table, so it is recorded verified=True by the same standard the
    # eleven are. Rows are brought to this corpus's section granularity
    # ("303(2)" -> "303", see _pair_confirmed) and kept only when BOTH keys
    # exist in the corpus -- a row naming a section this scrape does not have
    # would give V3/R2 a successor the verifier cannot itself check.
    # Checked against 28 independently known correspondences: 25 present and
    # correct, 3 absent from the table, none contradicted.
    known_sections = {(t[0], t[1]) for t in sections}
    seen = {(old, new) for old, new, _ in mapping}
    for (new_act, old_act), rows in all_rows.items():
        for r in rows:
            old_num = r.old_section.split("(", 1)[0].strip()
            new_num = r.new_section.split("(", 1)[0].strip()
            if ((old_act, old_num) not in known_sections
                    or (new_act, new_num) not in known_sections):
                continue
            pair = (f"{old_act}§{old_num}", f"{new_act}§{new_num}")
            if pair in seen:
                continue
            seen.add(pair)
            mapping.append((pair[0], pair[1], True))

    status_notes: Dict[str, str] = {}  # none identified in this scrape yet

    return sections, mapping, status_notes


def build_real_db(path: str = ":memory:"):
    """Same contract as data.seed_corpus.build_seed_db, over the real corpus."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.metadata_db import MetadataDB
    sections, mapping, status_notes = build_sections_and_mapping()
    db = MetadataDB(path)
    db.load_seed(sections, mapping, status_notes)
    return db


if __name__ == "__main__":
    sections, mapping, status_notes = build_sections_and_mapping()
    by_act: Dict[str, int] = {}
    for act, *_ in sections:
        by_act[act] = by_act.get(act, 0) + 1
    print(f"{len(sections)} real provisions across {len(by_act)} acts:")
    for act, n in sorted(by_act.items()):
        print(f"  {act:6} {n}")
    verified = sum(1 for _, _, v in mapping if v)
    print(f"\n{verified}/{len(mapping)} mapping entries verified against the "
          f"UP Police correspondence table series:")
    for _, _, _, url in CORRESPONDENCE_SOURCES:
        print(f"  {url}")
