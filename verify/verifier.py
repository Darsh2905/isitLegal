"""
Deterministic citation verifier (contribution C1).

Model-free post-processor. Performs no inference and makes no probabilistic
judgement, and therefore cannot itself hallucinate. Given a generated answer
and the retrieved chunks that were supplied to the generator, it enforces:

  V1  Every inline [n] marker indexes a chunk that was actually retrieved.
  V2  Every legal reference resolves to a provision present in the metadata DB.
  V3  Every cited provision carries current repeal / validity status; superseded
      provisions are annotated with their successor.
  V4  Every quoted span occurs verbatim (token for token) in a retrieved
      source, or is a faithful marked elision of one.

V2 covers three citation forms: statutory references (with an act named, or
bare and resolved from context), case names, and Supreme Court reporter
citations (the last two against the judgment corpus). A fail-closed residual
scan (verify/residual.py) strips any reference-shaped text the parser did not
claim, so a citation the parser cannot read is removed rather than passed.

Claims failing V1, V2 or V4 are stripped from the answer. The result is that
citation RESOLUTION on the emitted answer is 1.0 by construction: every
surviving marker, provision and case resolves to the corpus and every
surviving quotation is verbatim. It is not a guarantee that the cited source
SUPPORTS the claim -- that is V5's question, which is advisory and measured
separately.

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
from ingest.normalize import (extract_citations, extract_unresolved_act_refs,
                              extract_bare_references, extract_act_mentions,
                              Citation)
from ingest.case_citations import (CaseIndex, AttestedCases,
                                   extract_case_names,
                                   extract_reporter_citations)
from verify.residual import residual_references
from ingest.metadata_db import MetadataDB
from verify.repair import repair_markers, repair_superseded, RepairLog
from verify.support import SupportScreen


class Violation(Enum):
    UNRESOLVED_MARKER = "unresolved_marker"      # V1
    NONEXISTENT_PROVISION = "nonexistent_provision"  # V2
    UNKNOWN_ACT = "unknown_act"                  # V2: act named is not in the corpus
    UNPLACED_REFERENCE = "unplaced_reference"    # V2: bare section, no act in context
    UNKNOWN_CASE = "unknown_case"                # V2: case name/reporter not in corpus
    CASE_MISMATCH = "case_mismatch"              # V2: name and reporter name different cases
    UNPARSED_REFERENCE = "unparsed_reference"    # fail-closed residual scan
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
    # Fail-closed residual scan: reference-shaped spans no extractor claimed.
    residual_count: int = 0
    case_citations_total: int = 0
    case_citations_resolved: int = 0
    bare_references_total: int = 0

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
# Single capital letters are initials ("K. P. D. v. State", "Anvar P.V. v.
# P.K. Basheer"): splitting at one cut a case name in two, so neither half was
# read as a case citation and both passed (found by the round-2 audit).
_ABBREVS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [
    "Sec", "sec", "Secs", "secs", "Ss", "ss", "s", "No", "no", "Nos",
    "Art", "art", "Arts", "arts", "cl", "Cl", "para", "Para", "r", "R",
    "v", "vs", "Vs", "i.e", "e.g", "viz", "cf", "etc", "Ltd", "Pvt",
    "Anr", "Ors", "Hon", "Mr", "Mrs", "Ms", "Dr", "Jr", "Sr",
    "Cr", "St", "Co", "Govt", "Rs", "Retd", "retd", "Jt", "Smt", "Shri",
]
_DOT = "\u0000DOT\u0000"
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ABBREVS, key=len, reverse=True)) + r")\."
)

# A full stop right before the "v." of a case name never ends a sentence:
# "Puttaswamy (Retd.) v. Union of India".
_BEFORE_V = re.compile(r"\.(?=[)\]\"'\u201d\u2019*]*\s+(?:v|vs|versus)\.?\s)")


def _mask_abbrevs(text: str) -> str:
    text = _BEFORE_V.sub(_DOT, text)
    return _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, text)

def _unmask(text: str) -> str:
    return text.replace(_DOT, ".")

# Citation markers. "[n]" is the instructed form, but gpt-oss models also emit
# their native lenticular "\u3010n\u3011" (and full-width "\uff3bn\uff3d"): found in
# the live round, where V1 never saw those markers -- one pointing at no
# retrieved chunk would have passed. All three forms are markers.
_OPEN, _CLOSE = "[\\[\u3010\uff3b]", "[\\]\u3011\uff3d]"
_MARKER = re.compile(_OPEN + r"(\d+)" + _CLOSE)

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
_SENT_END = re.compile(r"[.!?][\"”’')]*(?:\s*" + _OPEN + r"\d+" + _CLOSE
                       + r")*(?=\s|$)")
_MARKERS_ONLY = re.compile(r"^[\s.,;:!?]*(?:" + _OPEN + r"\d+" + _CLOSE
                           + r"[\s.,;:!?]*)+$")


# A sentence boundary inside a quotation split the quotation in two: the
# first half was checked by V4 as an unterminated quote, and the second half
# -- with no opening mark -- was not checked at all. Statutory text is full of
# full stops ("...any records. information contained in..."), so a quotation
# of more than one sentence could carry a substitution in its tail past V4.
# Found by the 2026-10 quotation length sweep (eval/run_quote_eval.py).
# Sentence-final punctuation inside a quotation is masked before splitting.
_QUOTED = re.compile(r"[\"\u201c][^\"\u201d\n]*(?:[\"\u201d]|$)")
_QDOT = {".": "\u0000QP\u0000", "!": "\u0000QX\u0000", "?": "\u0000QQ\u0000"}


def _mask_quoted(text: str) -> str:
    return _QUOTED.sub(lambda m: re.sub(r"[.!?]", lambda d: _QDOT[d.group(0)],
                                        m.group(0)), text)


def _unmask_quoted(text: str) -> str:
    for ch, tok in _QDOT.items():
        text = text.replace(tok, ch)
    return text


def _segment(answer: str) -> List[tuple]:
    """Split an answer into (sentence, block_index) pairs."""
    masked = _LIST_NUM.sub(lambda m: m.group(1) + _DOT,
                           _mask_abbrevs(_mask_quoted(answer.strip())))
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
    return [(_unmask_quoted(_unmask(s)), b) for s, b in out]
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
    def __init__(self, db: MetadataDB, theta: float = 1.0,
                 repair: bool = False, support: bool = False,
                 tau_support: float = 0.20, graph=None,
                 cases: Optional[CaseIndex] = None, residual: bool = True):
        self.db = db
        # theta = 1.0: a quotation must occur verbatim, token for token. The
        # earlier theta = 0.95 on (longest contiguous run / quote length) was
        # length-dependent in a way its own justification missed: one
        # substituted word scores (n-1)/n when it sits at either END of an
        # n-token quote, so every quote of 20+ tokens could have its first or
        # last word changed and still pass (40+ tokens: its second word too).
        # The Article 21 probe that justified 0.95 is 18 tokens -- below the
        # length where the hole opens. Found by peer review (2026-10); see
        # eval/run_quote_eval.py's length sweep. Marked omissions ("...") are
        # handled by the separate elision rule, so exactness costs nothing a
        # faithful quotation needs.
        self.theta = theta
        # Case citations resolve against the judgment corpus. With no index
        # every case citation is unresolvable and is stripped (closed world).
        self.cases = cases if cases is not None else CaseIndex()
        self.residual = residual
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

    @staticmethod
    def _candidate_acts(marker_keys, citations, context_acts,
                        mentions=(), before=None, sentence=None) -> List[str]:
        # The act named nearest BEFORE the reference in the same sentence
        # first ("Const §22", "the CrPC ... Section 41") -- and if that
        # nearest name is an act outside the corpus ("the Indian Telegraph
        # Act, 1885 (§§ 5-7)") there is no candidate at all; then the acts
        # the sentence's markers point at; then acts cited earlier in the
        # sentence; then acts cited or named in earlier SURVIVING sentences.
        # Acts named after the reference are never used.
        acts = []
        if before is not None and sentence is not None:
            from ingest.normalize import act_names_before
            near = act_names_before(sentence, before)
            if near is not None:
                if near[0] is None:
                    return []
                acts.append(near[0])
        acts += [k.split("§", 1)[0] for k in marker_keys]
        acts += [c.act for c in citations
                 if before is None or c.start < before]
        acts += list(reversed(context_acts))
        out = []
        for a in acts:
            if a not in out:
                out.append(a)
        return out

    def _place_bare(self, b, marker_keys, citations, context_acts,
                    mentions=(), context_keys=(), sentence=None):
        """Resolve a bare reference to a Citation in the first candidate act
        where it exists, or None. A pinpoint-only reference ("sub-section
        (2)") is placed on a provision the sentence cites or points at."""
        if b.pinpoint_only:
            # The provision the sentence cites or points at; failing that,
            # the provision most recently cited in the answer ("Section 64
            # ... The text of sub-section (2) ..."). One candidate each: the
            # first that has the sub-division wins, nothing else is tried.
            bases = list(marker_keys) + [c.key for c in citations]
            if not bases and context_keys:
                bases = [context_keys[-1]]
            for base in bases:
                key = base.split("(", 1)[0] + b.section
                if self.db.resolve(key) is not None:
                    act, sec = key.split("§", 1)
                    return Citation(act=act, section=sec, raw=b.raw,
                                    start=b.start, end=b.end)
            return None
        cands = self._candidate_acts(marker_keys, citations, context_acts,
                                     mentions, before=b.start,
                                     sentence=sentence)
        # Only the FIRST candidate act is tried. Trying each in turn would
        # let "Section 7777" fall through to any act that happens to have a
        # section 7777 -- or let "Const §22" validate as BNSS 22. If the
        # nearest act does not have the section, the reference fails.
        for act in cands[:1]:
            if self.db.resolve(f"{act}§{b.section}") is not None:
                return Citation(act=act, section=b.section, raw=b.raw,
                                start=b.start, end=b.end)
        return None

    def verify(self, answer: str, retrieved: List[Chunk]) -> tuple:
        """
        Returns (verified_answer, VerificationReport).
        """
        report = VerificationReport()
        valid_indices = {c.index for c in retrieved}
        by_index = {c.index: c for c in retrieved}

        segments = _segment(answer)
        report.sentences_total = len(segments)
        attested = AttestedCases(c.text for c in retrieved)

        kept: List[tuple] = []  # (sentence, block_index)
        context_acts: List[str] = []  # acts cited so far, in order
        context_keys: List[str] = []  # provisions cited so far, in order

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
            marker_chunk_keys = [by_index[n].key for n in markers
                                 if n in by_index and "§" in by_index[n].key]

            # ---- bare references: resolve the act from context ----
            # "Section 41 permits arrest [3]": no act named. Placed in the
            # act of a chunk the sentence's markers point at, else the act
            # most recently cited in the answer. A bare reference that cannot
            # be placed in an act, or does not exist in the act it is placed
            # in, is stripped -- before this, it was never checked at all.
            mentions = extract_act_mentions(sent)
            bare = extract_bare_references(sent)
            report.bare_references_total += len(bare)
            unplaced = []
            for b in bare:
                placed = self._place_bare(b, marker_chunk_keys, citations,
                                          context_acts, mentions, context_keys,
                                          sentence=sent)
                if placed is None:
                    unplaced.append(b)
                else:
                    citations.append(placed)
            citations.sort(key=lambda c: c.start)

            # ---- case names and reporter citations ----
            case_names = extract_case_names(sent)
            reporters = extract_reporter_citations(sent)
            report.case_citations_total += len(case_names) + len(reporters)

            n_cites = (len(markers) + len(citations) + len(unresolved_acts)
                       + len(unplaced) + len(case_names) + len(reporters))
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

            for b in unplaced:
                report.findings.append(Finding(
                    Violation.UNPLACED_REFERENCE,
                    f"{b.raw!r}: no act names it, and it does not exist in any "
                    f"act in context ({', '.join(self._candidate_acts(marker_chunk_keys, citations, context_acts, mentions, b.start, sent)[:1]) or 'none'})",
                    sentence_index=i, span=b.raw))
                strip = True

            # ---- V2 for case law: names and reporters must resolve ----
            resolved_docs = set()
            for cn in case_names:
                doc = self.cases.resolve_name(cn)
                if doc is None and attested.has_name(cn):
                    doc = "attested"
                if doc is None:
                    report.findings.append(Finding(
                        Violation.UNKNOWN_CASE,
                        f"case {cn.raw!r} not found in the judgment corpus",
                        sentence_index=i, span=cn.raw))
                    strip = True
                else:
                    report.case_citations_resolved += 1
                    resolved_docs.add(("name", doc))
            for rc in reporters:
                doc = self.cases.resolve_reporter(rc)
                if doc is None and attested.has_reporter(rc):
                    doc = "attested"
                if doc is None:
                    report.findings.append(Finding(
                        Violation.UNKNOWN_CASE,
                        f"reporter citation {rc.raw!r} not found in the "
                        f"judgment corpus", sentence_index=i, span=rc.raw))
                    strip = True
                else:
                    report.case_citations_resolved += 1
                    resolved_docs.add(("rep", doc))
            name_docs = {d for k, d in resolved_docs
                         if k == "name" and d != "attested"}
            rep_docs = {d for k, d in resolved_docs
                        if k == "rep" and d != "attested"}
            if name_docs and rep_docs and not (name_docs & rep_docs):
                # A real case name with another real case's citation.
                report.findings.append(Finding(
                    Violation.CASE_MISMATCH,
                    "case name and reporter citation resolve to different "
                    "judgments", sentence_index=i))
                strip = True

            # ---- fail-closed residual scan ----
            if self.residual:
                covered = ([(c.start, c.end) for c in citations]
                           + [(r.start, r.end) for r in unresolved_acts]
                           + [(b.start, b.end) for b in bare]
                           + [(c.start, c.end) for c in case_names]
                           + [(r.start, r.end) for r in reporters])
                for r in residual_references(sent, covered):
                    report.findings.append(Finding(
                        Violation.UNPARSED_REFERENCE,
                        f"reference-shaped text the parser could not read "
                        f"({r.kind}): {r.raw!r} -- stripped, not passed",
                        sentence_index=i, span=r.raw))
                    report.residual_count += 1
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
            marker_keys = list(marker_chunk_keys)
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
                    # No retrieved verbatim text (ungrounded generation, or a
                    # paraphrased chunk): check the quote against the corpus
                    # text of the provisions the sentence itself cites.
                    for k in cited_keys:
                        prov = self.db.get(k)
                        if prov is not None and prov.verbatim:
                            checkable.append(Chunk(0, k, prov.text, True))

                if not checkable:
                    # FAIL CLOSED. This used to record a skip and PASS the
                    # quotation -- the same fail-open shape as a citation the
                    # parser cannot read: a quotation nothing can be checked
                    # against was emitted as if checked. It is now stripped.
                    report.quote_checks_skipped += 1
                    report.findings.append(Finding(
                        Violation.UNSUPPORTED_QUOTE,
                        "quotation has no verbatim source to check against "
                        "-- stripped, not passed", sentence_index=i, span=q))
                    strip = True
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
                    f"quoted span not verbatim in any cited source (longest "
                    f"verbatim run {best:.1f}% of the quote; "
                    f"{self.theta*100:.0f}% required)"
                    + (" (elided segments not all verbatim-in-order, or an "
                       "omission removes a negation/exception/condition)"
                       if _ELLIPSIS.search(q) else ""),
                    sentence_index=i, span=q))
                strip = True

            # Context for later bare references comes only from sentences
            # that SURVIVE: placing "Section 138" by an act named in a
            # stripped sentence left the emitted answer citing "Section 138"
            # with no act anywhere in it (found by the round-2 audit).
            if not strip:
                for c in citations:
                    context_acts.append(c.act)
                    context_keys.append(c.key)
                context_acts.extend(a for a, _ in mentions)

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
