"""
Deterministic citation verifier (contribution C1).

Model-free post-processor. Performs no inference and makes no probabilistic
judgement, and therefore cannot itself hallucinate. Given a generated answer
and the retrieved chunks that were supplied to the generator, it enforces:

  V1  Every inline [n] marker indexes a chunk that was actually retrieved.
  V2  Every legal reference resolves to a provision present in the metadata DB.
  V3  Every cited provision carries current repeal / validity status; superseded
      provisions are annotated with their successor.
  V4  Every quoted span fuzzy-matches a retrieved source span at >= theta.

Claims failing V1, V2 or V4 are stripped from the answer. The result is that
citation precision on the emitted answer is 1.0 by construction: not a measured
outcome but an invariant of the procedure.

The cost is coverage -- stripping reduces answer completeness -- so the
verifier reports what it removed, and evaluation must report citation RECALL
and coverage alongside precision.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from rapidfuzz import fuzz

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.normalize import extract_citations, extract_unresolved_act_refs, Citation
from ingest.metadata_db import MetadataDB
from verify.repair import repair_markers, repair_superseded, RepairLog
from verify.support import SupportScreen


class Violation(Enum):
    UNRESOLVED_MARKER = "unresolved_marker"      # V1
    NONEXISTENT_PROVISION = "nonexistent_provision"  # V2
    UNKNOWN_ACT = "unknown_act"                  # V2: act named is not in the corpus
    REPEALED_PROVISION = "repealed_provision"    # V3 (annotate, not strip)
    INVALIDATED_PROVISION = "invalidated_provision"  # V3 (annotate)
    UNSUPPORTED_QUOTE = "unsupported_quote"      # V4
    WEAK_SUPPORT = "weak_support"                # V5 (advisory, never strips)
    SUPPORT_VIA_REFERENCE = "support_via_reference"  # V5 (advisory, never strips)
    MARKER_MISMATCH = "marker_mismatch"          # advisory, never strips

# Findings that annotate rather than strip.
_ADVISORY = {"repealed_provision", "invalidated_provision", "weak_support",
             "support_via_reference", "marker_mismatch"}


@dataclass
class Finding:
    violation: Violation
    detail: str
    sentence_index: Optional[int] = None
    span: Optional[str] = None


@dataclass
class VerificationReport:
    findings: List[Finding] = field(default_factory=list)
    sentences_total: int = 0
    sentences_stripped: int = 0
    citations_total: int = 0
    citations_removed: int = 0
    annotations: List[str] = field(default_factory=list)
    quote_checks_performed: int = 0
    quote_checks_skipped: int = 0
    repair_log: "RepairLog" = field(default_factory=RepairLog)
    sentences_repaired: int = 0
    weak_support_count: int = 0
    support_via_reference_count: int = 0
    # Quotations that failed the strict contiguous-run test but passed the
    # ellipsis-aware test (every segment verbatim, in order, nothing
    # meaning-bearing omitted) -- counted so an accepted elision is never
    # invisible in the report.
    quotes_accepted_with_elision: int = 0
    marker_mismatch_count: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of generated sentences surviving verification."""
        if self.sentences_total == 0:
            return 1.0
        return 1.0 - (self.sentences_stripped / self.sentences_total)

    def by_type(self, v: Violation) -> List[Finding]:
        return [f for f in self.findings if f.violation == v]

    def has_violations(self) -> bool:
        return any(f.violation.value not in _ADVISORY for f in self.findings)


@dataclass
class Chunk:
    """A retrieved source chunk supplied to the generator."""
    index: int          # the [n] the generator may cite
    key: str            # canonical provision key, e.g. IPC§420
    text: str
    verbatim: bool      # is `text` real statutory language?


# Legal prose is dense with abbreviations ending in a period. Splitting naively
# on [.!?] shatters "charged under Sec. 420 IPC." into "Sec." and "420 IPC.",
# severing a citation from the claim it supports -- which silently defeats V2,
# because the orphaned fragment no longer parses as a citation. Abbreviation
# periods are therefore masked before splitting and restored afterwards.
_ABBREVS = [
    "Sec", "sec", "Secs", "secs", "Ss", "ss", "s", "S", "No", "no", "Nos",
    "Art", "art", "Arts", "arts", "cl", "Cl", "para", "Para", "r", "R",
    "v", "vs", "Vs", "i.e", "e.g", "viz", "cf", "etc", "Ltd", "Pvt",
    "Anr", "Ors", "Hon", "Mr", "Mrs", "Ms", "Dr", "Jr", "Sr",
    "Cr", "P", "C", "I", "N", "A", "B", "U", "St", "Co", "Govt", "Rs",
]
_DOT = "\u0000DOT\u0000"
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREVS, key=len, reverse=True)) + r")\."
)

def _mask_abbrevs(text: str) -> str:
    return _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, text)

def _unmask(text: str) -> str:
    return text.replace(_DOT, ".")

_MARKER = re.compile(r"\[(\d+)\]")

# SENTENCE SEGMENTATION. The original splitter cut only at sentence-final
# punctuation followed by whitespace, and real generated output (a live
# gpt-oss-20b answer set, eval/live_generation_answers.py) broke it two ways:
#   - "...liable to fine. [1][2]" -- markers AFTER the full stop became their
#     own "sentence", leaving the claim itself uncited: V5 never screened it,
#     V4 checked its quotes against every chunk instead of the cited ones, and
#     had [1] been invalid only the bare "[1][2]" fragment would have been
#     stripped while the claim survived.
#   - "...caused it.[1]  \n\nThe same definition ... section 80.[2]" -- a
#     marker glued to the full stop with no space, so NOTHING split: several
#     claims were verified, and would have been stripped, as one unit, and
#     V5's best-of-all-keys verdict let one well-supported claim mask another.
# Segmentation now (1) treats line breaks as boundaries (bullets, numbered
# lists, paragraphs), (2) ends a sentence after any markers that trail its
# final punctuation, and (3) folds a markers-only fragment back into the
# sentence before it. Leading list numbering ("1. ") is masked like an
# abbreviation so it does not end a sentence by itself.
_LIST_NUM = re.compile(r"(?m)^(\s*(?:[-*•]\s*)?\d{1,3})\.(?=\s)")
_SENT_END = re.compile(r"[.!?][\"”’')]*(?:\s*\[\d+\])*(?=\s|$)")
_MARKERS_ONLY = re.compile(r"^[\s.,;:!?]*(?:\[\d+\][\s.,;:!?]*)+$")


def _segment(answer: str) -> List[tuple]:
    """Split an answer into (sentence, block_index) pairs."""
    masked = _LIST_NUM.sub(lambda m: m.group(1) + _DOT,
                           _mask_abbrevs(answer.strip()))
    out: List[list] = []
    for b, block in enumerate(masked.split("\n")):
        block = block.strip()
        if not block:
            continue
        pieces, pos = [], 0
        for m in _SENT_END.finditer(block):
            pieces.append(block[pos:m.end()])
            pos = m.end()
        pieces.append(block[pos:])
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if _MARKERS_ONLY.match(piece) and out:
                out[-1][0] = out[-1][0] + " " + piece
                continue
            out.append([piece, b])
    return [(_unmask(s), b) for s, b in out]
# Non-greedy, and tolerant of an unclosed quotation: an unterminated quote
# previously matched nothing at all, so a fabricated quotation missing its
# closing mark bypassed V4 entirely.
_QUOTE = re.compile(r"[\"\u201c](.{8,}?)(?:[\"\u201d]|$)", re.DOTALL)
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str):
    return _TOKEN.findall(s.lower())


def _quote_score(quote: str, source: str) -> float:
    """
    Token-level verbatim score in [0, 100].

    Character-level fuzzy matching is unsuitable for legal quotation checking:
    substituting "statute" for "law" in an eighteen-word quotation scores 95.9%
    by character ratio, yet inverts the meaning of the provision. A verbatim
    quotation must be verbatim, so we score at TOKEN granularity:

        score = (longest contiguous run of quote tokens found in the source)
                / (number of quote tokens) * 100

    An exact quotation, modulo case, punctuation and whitespace, scores 100.
    A single substituted word in an n-token quote caps the score below
    (n-1)/n * 100, so the threshold rejects it.
    """
    from difflib import SequenceMatcher
    q_tok, s_tok = _tokens(quote), _tokens(source)
    if not q_tok:
        return 0.0
    m = SequenceMatcher(a=q_tok, b=s_tok, autojunk=False)
    block = m.find_longest_match(0, len(q_tok), 0, len(s_tok))
    return 100.0 * block.size / len(q_tok)


# ELIDED QUOTATIONS. The contiguous-run score above rejects every quotation
# that omits words with an explicit ellipsis, even a perfectly faithful one:
# of the eight quotations V4 stripped from the live gpt-oss-20b answers,
# seven were verbatim segments in order with the omission marked "..." --
# e.g. "any person who...is an enemy alien" (omitting "for the time being").
# Marked omission is standard legal quotation practice, not misquotation.
#
# It is not harmless in general, though: an ellipsis can delete the one word
# that carries a provision's legal effect. Two of those seven omitted an
# exception -- CrPC s.57's "in the absence of a special order of a
# magistrate under section 167" and Art. 22's "excluding the time necessary
# for the journey" -- and "shall ... be deprived" from "shall not be
# deprived" inverts a provision outright. An elided quotation is therefore
# accepted only when ALL of the following hold, and fails otherwise:
#   - every ellipsis-separated segment occurs verbatim (token level) in ONE
#     retrieved chunk, in the quoted order;
#   - each omitted stretch is at most _MAX_ELIDED_TOKENS long (no stitching
#     together fragments from distant parts of a long provision);
#   - no omitted stretch contains a word that negates, excepts, conditions,
#     or alternates/conjoins the retained text (_MEANING_CUES).
# The cue list is deliberately broad ("and"/"or" included): a wrongly
# rejected faithful elision costs coverage, a wrongly accepted one costs the
# precision guarantee, and only the second is unacceptable here.
_ELLIPSIS = re.compile(r"…|\.\s?\.\s?\.")
_MAX_ELIDED_TOKENS = 40
_MEANING_CUES = frozenset({
    "not", "no", "never", "nothing", "none", "neither", "nor", "cannot",
    "except", "excepting", "exception", "unless", "save", "saving",
    "without", "excluding", "exclusive", "provided", "proviso",
    "notwithstanding", "subject", "only", "but", "absence", "otherwise",
    "until", "other", "and", "or",
})


def _elided_quote_ok(quote: str, source: str):
    """Returns (accepted, [omitted stretches]) for a quote containing an
    explicit ellipsis; (False, []) if it has fewer than two segments."""
    segs = [t for t in (_tokens(s) for s in _ELLIPSIS.split(quote)) if t]
    if len(segs) < 2:
        return False, []
    st = _tokens(source)

    def occurrences(seg, start):
        n = len(seg)
        return [i for i in range(start, len(st) - n + 1) if st[i:i + n] == seg]

    for first in occurrences(segs[0], 0):
        pos, omitted, ok = first + len(segs[0]), [], True
        for seg in segs[1:]:
            nxt = occurrences(seg, pos)
            if not nxt:
                ok = False
                break
            gap = st[pos:nxt[0]]
            if (len(gap) > _MAX_ELIDED_TOKENS
                    or any(t in _MEANING_CUES for t in gap)):
                ok = False
                break
            omitted.append(" ".join(gap))
            pos = nxt[0] + len(seg)
        if ok:
            return True, omitted
    return False, []


class CitationVerifier:
    def __init__(self, db: MetadataDB, theta: float = 0.95,
                 repair: bool = False, support: bool = False,
                 tau_support: float = 0.20, graph=None):
        self.db = db
        self.theta = theta
        # V5 is ADVISORY ONLY and never strips. Its score distributions for
        # correct and topically-adjacent-incorrect attributions overlap, so any
        # threshold that catches adjacent misattribution also deletes correct
        # claims. tau_support is therefore set at the highest value that keeps
        # flagging precision at 1.0, trading recall for a guarantee that no
        # correct claim is ever flagged.
        # An optional statutory citation graph lets V5 distinguish a claim that
        # is unsupported from one whose cited provision is merely not
        # self-contained. Deterministic; see SupportScreen.resolve.
        self.support = (SupportScreen(db, tau_support, graph=graph)
                        if support else None)
        # repair=False reproduces the strip-only baseline, which is retained
        # as the ablation anchor. Repair never relaxes a check; it re-runs the
        # same checks on the repaired sentence.
        self.repair = repair

    def _related(self, a: str, b: str) -> bool:
        """Whether two provisions are the same, recodification counterparts,
        or joined by an express cross-reference -- i.e. whether a marker
        pointing at one is a coherent attribution for a claim citing the
        other."""
        if a == b:
            return True
        if b in self.db.successor(a) or a in self.db.successor(b):
            return True
        graph = self.support.graph if self.support is not None else None
        if graph is not None and (b in graph.outbound(a) or a in graph.outbound(b)):
            return True
        return False

    def verify(self, answer: str, retrieved: List[Chunk]) -> tuple:
        """
        Returns (verified_answer, VerificationReport).
        """
        report = VerificationReport()
        valid_indices = {c.index for c in retrieved}
        by_index = {c.index: c for c in retrieved}

        segments = _segment(answer)
        report.sentences_total = len(segments)

        kept: List[tuple] = []  # (sentence, block_index)

        for i, (sent, block) in enumerate(segments):
            strip = False

            # ---- repair pass (optional, runs BEFORE verification) ----
            # Each rule is admissible only if the repaired sentence then passes
            # the same checks below. Repair therefore cannot trade precision
            # for coverage.
            if self.repair:
                report.repair_log.attempted += 1
                repaired_any = False
                fixed, rep = repair_markers(sent, i, retrieved, self.db,
                                            related=self._related)
                if rep is not None:
                    sent = fixed; report.repair_log.add(rep); repaired_any = True
                fixed, rep = repair_superseded(sent, i, self.db)
                if rep is not None:
                    sent = fixed; report.repair_log.add(rep); repaired_any = True
                if repaired_any:
                    report.sentences_repaired += 1

            citations = extract_citations(sent)
            unresolved_acts = extract_unresolved_act_refs(sent)
            markers = [int(m) for m in _MARKER.findall(sent)]
            n_cites = len(markers) + len(citations) + len(unresolved_acts)
            report.citations_total += n_cites

            # ---- V1: markers must index a retrieved chunk ----
            for n in markers:
                if n not in valid_indices:
                    report.findings.append(Finding(
                        Violation.UNRESOLVED_MARKER,
                        f"marker [{n}] does not index any retrieved chunk",
                        sentence_index=i))
                    strip = True

            # ---- V2: an act the corpus does not contain cannot be checked ----
            for ref in unresolved_acts:
                report.findings.append(Finding(
                    Violation.UNKNOWN_ACT,
                    f"section {ref.section} of {ref.act_name!r} (from "
                    f"{ref.raw!r}): act not in corpus, citation cannot be "
                    f"verified", sentence_index=i, span=ref.raw))
                strip = True

            # ---- V2/V3: legal references must exist; check status ----
            cited_keys: List[str] = []
            for cit in citations:
                # resolve() falls back from a sub-section pinpoint to its
                # section only when the sub-section labels occur, in order,
                # in that section's own text (MetadataDB.resolve).
                prov = self.db.resolve(cit.key)
                if prov is None:
                    report.findings.append(Finding(
                        Violation.NONEXISTENT_PROVISION,
                        f"{cit.key} (from {cit.raw!r}) not found in corpus",
                        sentence_index=i, span=cit.raw))
                    strip = True
                    continue
                if prov.key not in cited_keys:
                    cited_keys.append(prov.key)

                if prov.status_note:
                    report.findings.append(Finding(
                        Violation.INVALIDATED_PROVISION,
                        f"{prov.key}: {prov.status_note}",
                        sentence_index=i, span=cit.raw))
                    report.annotations.append(
                        f"Note: {prov.key} — {prov.status_note}")
                elif prov.repealed_by is not None:
                    succ = self.db.successor(prov.key)
                    if len(succ) == 1:
                        s = succ[0]
                        verified = self.db.is_mapping_verified(prov.key, s)
                        qual = "" if verified else " (correspondence pending verification)"
                        msg = (f"Note: {prov.key} was superseded by the "
                               f"{prov.repealed_by}; the corresponding current "
                               f"provision is {s}.{qual}")
                    elif succ:
                        # A section the recodification SPLIT (IPC 498A ->
                        # BNS 85 and 86): name every counterpart rather than
                        # silently picking one.
                        msg = (f"Note: {prov.key} was superseded by the "
                               f"{prov.repealed_by}; its content is now split "
                               f"across {', '.join(succ)}.")
                    else:
                        msg = (f"Note: {prov.key} was superseded by the "
                               f"{prov.repealed_by}; no mapping recorded.")
                    report.findings.append(Finding(
                        Violation.REPEALED_PROVISION, msg,
                        sentence_index=i, span=cit.raw))
                    report.annotations.append(msg)

            # ---- marker/citation consistency (advisory) ----
            # "Article 21 protects personal liberty [1]" with [1] = Article 19
            # passed every check: the marker resolves (V1), Article 21 exists
            # (V2), and V5's best-of-all-cited-keys verdict let the correct
            # textual citation mask the wrong pointer. When a sentence names
            # provisions explicitly AND carries markers, at least one marker
            # should point at a named provision, its recodification
            # counterpart, or a provision it expressly cross-references.
            # Advisory, not stripping: the claim itself may still be sound,
            # only its pointer is wrong (repair rule R3 repoints it when the
            # named provision was retrieved).
            marker_keys = [by_index[n].key for n in markers if n in by_index]
            if cited_keys and marker_keys and not any(
                    self._related(c, mk) for c in cited_keys for mk in marker_keys):
                report.findings.append(Finding(
                    Violation.MARKER_MISMATCH,
                    f"sentence cites {', '.join(cited_keys)} but its marker(s) "
                    f"point at {', '.join(marker_keys)}",
                    sentence_index=i))
                report.marker_mismatch_count += 1

            # ---- V5: advisory topical-support screen ----
            # Runs before V4 so that a flagged-but-surviving sentence still
            # carries its advisory note. Never sets `strip`.
            if self.support is not None:
                # A claim may be attributed either by an explicit textual
                # citation ("Section 138 NI Act") or by a bare marker ("[1]"),
                # and generators routinely do the latter. Screening only the
                # former leaves marker-attributed claims -- the common case --
                # entirely unscreened, so both sources are collected.
                keys = list(cited_keys)
                keys += [k for k in marker_keys if k not in keys]
                if keys:
                    verdict, best = self.support.resolve_best(sent, keys)
                    if verdict == SupportScreen.WEAK:
                        report.findings.append(Finding(
                            Violation.WEAK_SUPPORT,
                            f"cited provision shows weak topical support for "
                            f"this claim ({best.explain()})",
                            sentence_index=i))
                        report.weak_support_count += 1
                    elif verdict == SupportScreen.VIA_REFERENCE:
                        # Not silently passed: the cited provision's own text
                        # does not support the claim, and we say so, naming the
                        # provision it refers to that does.
                        report.findings.append(Finding(
                            Violation.SUPPORT_VIA_REFERENCE,
                            f"cited provision is not self-contained; support "
                            f"is carried by an express cross-reference "
                            f"({best.explain()})",
                            sentence_index=i))
                        report.support_via_reference_count += 1

            # ---- V4: quoted spans must match a retrieved source ----
            for q in _QUOTE.findall(sent):
                cited_chunks = [by_index[n] for n in markers if n in by_index]
                pool = cited_chunks or retrieved
                checkable = [c for c in pool if c.verbatim]

                if not checkable:
                    # Cannot verify a quote against paraphrased text; record
                    # the skip rather than silently passing it.
                    report.quote_checks_skipped += 1
                    continue

                report.quote_checks_performed += 1
                best = max(
                    (_quote_score(q, c.text) for c in checkable), default=0.0)
                if best >= self.theta * 100:
                    continue
                elided = [_elided_quote_ok(q, c.text) for c in checkable]
                if any(ok for ok, _ in elided):
                    report.quotes_accepted_with_elision += 1
                    continue
                report.findings.append(Finding(
                    Violation.UNSUPPORTED_QUOTE,
                    f"quoted span best match {best:.1f}% < "
                    f"{self.theta*100:.0f}% threshold"
                    + (" (elided segments not all verbatim-in-order, or an "
                       "omission removes a negation/exception/condition)"
                       if _ELLIPSIS.search(q) else ""),
                    sentence_index=i, span=q))
                strip = True

            if strip:
                report.sentences_stripped += 1
                report.citations_removed += n_cites
            else:
                kept.append((sent, block))

        # Re-assemble, preserving the answer's line structure (bullets,
        # paragraphs) rather than flattening everything onto one line.
        lines: List[str] = []
        last_block = None
        for sent, block in kept:
            if block == last_block:
                lines[-1] += " " + sent
            else:
                lines.append(sent)
                last_block = block
        verified_answer = "\n".join(lines)
        # Deduplicate annotations, preserving order.
        seen = set()
        anns = [a for a in report.annotations
                if not (a in seen or seen.add(a))]
        report.annotations = anns
        if anns:
            verified_answer += "\n\n" + "\n".join(anns)

        if not kept:
            verified_answer = ("Insufficient verifiable authority: all generated "
                               "claims failed citation verification.\n\n" +
                               "\n".join(anns))

        return verified_answer, report
