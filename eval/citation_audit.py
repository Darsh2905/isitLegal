"""
Independent citation audit for generated answers, shared by
eval/run_generation_eval.py, eval/run_live_generation_eval.py and
eval/run_live_round2_eval.py.

Re-derives, from the answer text alone, how many citations it carries and how
many fail to resolve -- it does not read the verifier's report (the audit()
pattern of eval/run_repair_eval.py). Counted, since the 2026-10 fail-closed
revision, in every form the verifier now checks:

  statutory citations with an act named      resolve in the metadata DB
  sections cited to an act not in the corpus  never resolve
  bare sections ("Section 41", no act)        resolve if the section exists in
                                              some act the answer names or
                                              cites before it, or a cited
                                              chunk belongs to
  case names and SC reporter citations        resolve in the judgment corpus,
                                              or occur in a retrieved chunk
  markers [n], 【n】, ［n］              index a retrieved chunk
  reference-shaped text no extractor reads    never resolves (cannot be shown to)

Kept separate from "bad", as before: citations to real but REPEALED law with
no recorded successor, which V3 annotates by design rather than strips.

The bare-section rule here is deliberately looser than the verifier's (which
tries only the nearest act): the audit asks whether anything unresolvable
survived, and a looser "resolves" can only make the audit report fewer
failures than the verifier would, never hide a stricter failure the
verifier missed -- if the post-verifier audit is ever below 1.000, the
verifier has let something through.
"""
import re

from ingest.normalize import (extract_citations, extract_unresolved_act_refs,
                              extract_bare_references, extract_act_mentions)
from ingest.case_citations import (extract_case_names,
                                   extract_reporter_citations, AttestedCases)
from verify.residual import residual_references

_MARK = re.compile(r"[\[【［](\d+)[\]】］]")


def citation_stats(answer, chunks, db, cases):
    """(total, bad, unmapped_repealed, by_form) for one answer."""
    total = bad = unmapped = 0
    by_form = {}

    def count(form, ok):
        nonlocal total, bad
        total += 1
        bad += (not ok)
        t, b = by_form.get(form, (0, 0))
        by_form[form] = (t + 1, b + (not ok))

    cits = extract_citations(answer)
    for c in cits:
        prov = db.resolve(c.key)
        count("statutory", prov is not None)
        if prov is not None and prov.repealed_by is not None \
                and not db.successor(prov.key):
            unmapped += 1
    unres = extract_unresolved_act_refs(answer)
    for _ in unres:
        count("unknown act", False)

    marker_acts = {c.key.split("§", 1)[0] for c in chunks if "§" in c.key}
    mentions = extract_act_mentions(answer)
    bare = extract_bare_references(answer)
    for b in bare:
        if b.pinpoint_only:
            bases = [c.key for c in chunks if "§" in c.key] + [c.key for c in cits]
            ok = any(db.resolve(k.split("(", 1)[0] + b.section) for k in bases)
        else:
            acts = marker_acts | {a for a, pos in mentions if pos < b.start} \
                | {c.act for c in cits if c.start < b.start}
            ok = any(db.resolve(f"{a}§{b.section}") for a in acts)
        count("bare section", ok)

    names = extract_case_names(answer)
    reps = extract_reporter_citations(answer)
    attested = AttestedCases(c.text for c in chunks)
    for n in names:
        count("case name", cases.resolve_name(n) is not None
              or attested.has_name(n))
    for r in reps:
        count("reporter", cases.resolve_reporter(r) is not None
              or attested.has_reporter(r))

    valid = {c.index for c in chunks}
    for n in _MARK.findall(answer):
        count("marker", int(n) in valid)

    covered = ([(c.start, c.end) for c in cits] + [(r.start, r.end) for r in unres]
               + [(b.start, b.end) for b in bare]
               + [(n.start, n.end) for n in names] + [(r.start, r.end) for r in reps])
    for _ in residual_references(answer, covered):
        count("unreadable reference", False)
    return total, bad, unmapped, by_form


def strip_annotations(verified: str) -> str:
    """The verified answer without the verifier's appended V3 notes."""
    return verified.split("\n\nNote:")[0]
