"""
Verification repair (contribution C1b).

THE PROBLEM WITH STRIP-ONLY VERIFICATION
----------------------------------------
The base verifier attains citation precision 1.0 by deleting any claim whose
citation fails to resolve. That is sound but blunt: measured coverage on faulty
answers is 0.436, meaning well over half the generated content is discarded. A
reviewer is entitled to object that this is censorship rather than verification
-- precision 1.0 is trivially attainable by answering nothing.

Repair addresses this. Before discarding a claim, we ask whether the fault is
*recoverable from evidence already in hand*. Three faults are:

  R1  MARKER MISATTRIBUTION. The generator writes a substantively correct claim
      about a provision that WAS retrieved, but attaches the wrong marker --
      cites [9] when the supporting chunk is [3]. The claim is true and
      supported; only the pointer is wrong. If the provision named in the
      sentence matches a retrieved chunk, we repoint the marker at that chunk.

  R2  SUPERSEDED REFERENCE. The generator cites a repealed provision whose
      successor is recorded in the mapping table. Rather than merely appending
      a note, we rewrite the reference to the current provision and preserve
      the original in a parenthetical. Only a UNIQUE successor is written
      in; a section the recodification split across several is annotated
      (V3) with all of them rather than rewritten to one.

  R3  MARKER CONTRADICTS CITATION. Every marker resolves, but none points at
      the provision the sentence itself names (or its recodification
      counterpart, or a provision it expressly cross-references), while that
      provision WAS retrieved. Repointed at that chunk, on the same evidence
      standard as R1.

SOUNDNESS
---------
Repair must not weaken the invariant. Each rule is admissible only because it
terminates in the same check the strip path would have applied:

  R1 fires only when the provision named in the sentence is present in the
     retrieved set. The repaired marker therefore points at a chunk that
     genuinely concerns the cited provision, so the repaired sentence satisfies
     V1 by the same test any surviving sentence must pass.

  R2 fires only when the mapping table records exactly one successor that
     exists in the metadata DB. The rewritten reference therefore satisfies
     V2 and V3.

  R3 fires only when the provision the sentence explicitly names is in the
     retrieved set, so -- as with R1 -- the new pointer is verifiably right.

Anything not repairable by these rules is still stripped. Repair strictly
enlarges the set of surviving claims without admitting any claim that would
have failed verification -- it cannot trade precision for coverage, because
every repaired sentence is re-verified after repair.

Faults that are NOT repaired, by design: a fabricated provision (there is no
evidence from which to recover it) and a fabricated quotation (repairing it
would mean inventing text, which is precisely what the system exists to
prevent).
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.normalize import extract_citations
from ingest.metadata_db import MetadataDB


@dataclass
class Repair:
    rule: str            # "R1" | "R2" | "R3"
    sentence_index: int
    before: str
    after: str
    justification: str


@dataclass
class RepairLog:
    repairs: List[Repair] = field(default_factory=list)
    attempted: int = 0
    succeeded: int = 0

    def add(self, r: Repair):
        self.repairs.append(r)
        self.succeeded += 1


_MARKER = re.compile(r"\[(\d+)\]")


def repair_markers(sentence: str, idx: int, retrieved, db,
                   related=None) -> Tuple[str, Optional[Repair]]:
    """
    R1: repoint an unresolved marker at the chunk that actually supports it.

    Admissible only when a provision named in the sentence is present in the
    retrieved set, so the repaired marker is verifiably correct rather than
    merely plausible.
    """
    valid = {c.index for c in retrieved}
    markers = [int(n) for n in _MARKER.findall(sentence)]
    bad = [n for n in markers if n not in valid]

    key_to_index = {c.key: c.index for c in retrieved}
    index_to_key = {c.index: c.key for c in retrieved}
    # Resolve sub-section pinpoints to the section a chunk actually holds
    # ("Section 41(1)(b) CrPC" names retrieved chunk CRPC§41).
    cited = []
    for c in extract_citations(sentence):
        k = db.resolve_key(c.key) if db is not None else c.key
        if k and k not in cited:
            cited.append(k)
    supported = [k for k in cited if k in key_to_index]

    if not supported:
        return sentence, None  # nothing to repoint at; caller will strip

    target = key_to_index[supported[0]]

    if bad:
        out = sentence
        for n in bad:
            out = out.replace(f"[{n}]", f"[{target}]")
        return out, Repair(
            rule="R1", sentence_index=idx, before=sentence, after=out,
            justification=(f"marker(s) {bad} unresolved; sentence cites "
                           f"{supported[0]}, which is retrieved chunk "
                           f"[{target}] -- marker repointed"),
        )

    # R3: every marker resolves, but NONE points at any provision the
    # sentence itself names, while one of those provisions WAS retrieved --
    # "Article 21 protects personal liberty [1]" with [1] = Article 19 and
    # Article 21 retrieved as [2]. Same evidence standard as R1: the target
    # is the retrieved chunk holding the provision the sentence explicitly
    # cites, so the repaired pointer is verifiably right, and the sentence is
    # re-verified afterwards like any other.
    # `related` (supplied by the verifier) treats a marker pointing at a
    # cited provision's recodification counterpart or an express
    # cross-reference as coherent -- those are not repointed.
    rel = related or (lambda a, b: a == b)
    if markers and not any(rel(index_to_key[n], c)
                           for n in markers for c in cited):
        wrong = sorted(set(markers))
        out = sentence
        for n in wrong:
            out = out.replace(f"[{n}]", f"[{target}]")
        return out, Repair(
            rule="R3", sentence_index=idx, before=sentence, after=out,
            justification=(f"marker(s) {wrong} point at "
                           f"{', '.join(index_to_key[n] for n in wrong)}, not "
                           f"at the provision the sentence cites "
                           f"({supported[0]}, retrieved chunk [{target}]) -- "
                           f"marker repointed"),
        )
    return sentence, None


def repair_superseded(sentence: str, idx: int, db: MetadataDB) -> Tuple[str, Optional[Repair]]:
    """
    R2: rewrite a superseded reference to its current successor.

    Admissible only when the mapping table records a successor present in the
    metadata DB. The original reference is preserved parenthetically so the
    rewrite is transparent rather than silent.
    """
    out = sentence
    changed: List[str] = []

    cits = extract_citations(sentence)
    # Members of one enumeration ("sections 302 and 420 of the IPC") share a
    # single span. Splicing a replacement in once per member corrupted the
    # sentence -- "Section 318 BNS* (formerly sections 302 and 420 of the
    # IPC)ions 302 and 420 of the IPC)" -- and the garbled text then passed
    # re-verification, because every citation left in it still resolved. Found
    # by the fail-closed residual scan (verify/residual.py), which flagged the
    # orphaned "420 of the IPC" no extractor could read. An enumeration is now
    # left for V3, which annotates each member with its successor.
    span_count = {}
    for c in cits:
        span_count[(c.start, c.end)] = span_count.get((c.start, c.end), 0) + 1
    for cit in sorted(cits, key=lambda c: -c.start):  # right-to-left; offsets stable
        if span_count[(cit.start, cit.end)] > 1:
            continue
        prov = db.resolve(cit.key)
        if prov is None or prov.repealed_by is None:
            continue
        succ = db.successor(prov.key)
        # Rewrite only to a UNIQUE successor. Where the recodification split
        # a section (IPC 498A -> BNS 85 and 86), choosing one would be a
        # guess; V3's annotation names all of them instead.
        if len(succ) != 1 or not db.exists(succ[0]):
            continue
        new_key = succ[0]
        act, num = new_key.split("§")
        verified = db.is_mapping_verified(prov.key, new_key)
        flag = "" if verified else "*"

        # The matched span may begin with a connective ("under section 420
        # IPC", "u/s 302 IPC"). Dropping it produces ungrammatical output
        # ("is punishable Section 318 BNS"), so the leading connective is
        # detected and carried through to the rewritten reference.
        lead = ""
        m_lead = re.match(r"(?i)^(under\s+|u/s\s+)", cit.raw)
        if m_lead:
            lead = m_lead.group(1)

        replacement = (f"{lead}Section {num} {act}{flag} "
                       f"(formerly {cit.raw})")
        out = out[:cit.start] + replacement + out[cit.end:]
        changed.append(f"{cit.key}->{new_key}")

    if not changed:
        return sentence, None

    return out, Repair(
        rule="R2", sentence_index=idx, before=sentence, after=out,
        justification=("superseded reference(s) rewritten to current "
                       "provision: " + ", ".join(changed) +
                       "   (* = correspondence pending verification)"),
    )
