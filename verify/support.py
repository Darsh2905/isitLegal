"""
Deterministic support screening (check V5).

THE BLIND SPOT
--------------
Checks V1-V4 establish that a citation is *well-formed*: the marker resolves,
the provision exists, it is in force, and any quotation is verbatim. None of
them establishes that the cited provision has anything to do with the claim it
is attached to. The sentence

    "Murder is punishable under Section 138 of the NI Act [1]."

passes all four checks. Section 138 is real, in force, and correctly retrieved;
it simply concerns dishonoured cheques rather than murder. This is the failure
the paper's limitations section concedes, and it is the most damaging one,
because the output is maximally plausible: a genuine provision, correctly
formatted, attached to an unrelated proposition.

WHAT THIS MODULE DOES, AND WHAT IT DOES NOT
-------------------------------------------
Full entailment checking requires semantic judgement, which would reintroduce a
model into the verification layer and with it the possibility that the checker
hallucinates. We therefore do not attempt entailment. We implement a strictly
weaker but deterministic test: a NECESSARY-CONDITION SCREEN.

    If a provision genuinely supports a claim, the claim's substantive
    vocabulary should overlap the provision's own subject matter.

The converse does not hold -- overlap does not imply support -- so a passing
score is not evidence of correctness. The screen is therefore sound in one
direction only, and is reported as such: it detects topical mismatch, and is
silent about subtler misstatements of a topically correct provision.

SCORING
-------
Rare terms carry the signal. "Section", "punishable", "law" and "provided"
appear throughout a statutory corpus and discriminate nothing; "cheque",
"dowry", "murder" discriminate strongly. We therefore weight overlap by inverse
document frequency computed over the corpus itself, so the weighting adapts to
whatever collection the system is deployed on rather than relying on a fixed
stopword list.

    support(claim, provision) =
        sum of IDF over claim terms found in the provision
        --------------------------------------------------
        sum of IDF over all claim terms

Terms inside the citation span itself are excluded, since "section", "IPC" and
the section number would otherwise inflate the score for every claim.
"""

import math
import re
import sys
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.normalize import extract_citations
from ingest.metadata_db import MetadataDB

_WORD = re.compile(r"[a-z]{3,}")

# Structural vocabulary of statutory drafting. These are removed outright
# rather than down-weighted: in a corpus made entirely of statutes their IDF is
# low but nonzero, and retaining them adds noise without signal.
_STRUCTURAL = {
    "section", "sections", "act", "code", "sanhita", "adhiniyam", "article",
    "articles", "clause", "sub", "provided", "provision", "provisions",
    "punishable", "punishment", "punished", "shall", "may", "any", "the",
    "under", "this", "that", "which", "such", "with", "for", "and", "not",
    "law", "laws", "person", "whoever", "means", "case", "cases", "made",
    "other", "otherwise", "specified", "referred", "respect", "manner",
    "extend", "term", "liable", "offence", "offences", "court", "india",
    "indian", "bharatiya", "matter", "there", "been", "have", "has",
    # Auxiliaries and claim-framing verbs. These describe the ACT OF ASSERTING
    # rather than the subject matter asserted, and never carry topical signal:
    # a statute does not contain the word "provides" in a way that helps
    # distinguish it from any other statute.
    "was", "were", "are", "is", "be", "being", "will", "would", "can",
    "could", "must", "said", "states", "provides", "deals", "covers",
    "applies", "arises", "concerns", "governed", "framed", "filed",
    "charged", "charges", "relevant", "applicable", "according",
}


@dataclass
class SupportScore:
    score: float
    matched: List[str]
    missed: List[str]
    provision_key: str
    # Set when the cited provision's own text does not support the claim but a
    # provision it explicitly refers to does. See SupportScreen.resolve.
    via_key: Optional[str] = None
    via_score: float = 0.0

    def explain(self) -> str:
        m = ", ".join(self.matched[:4]) or "none"
        x = ", ".join(self.missed[:4]) or "none"
        base = (f"{self.provision_key}: score={self.score:.2f} "
                f"matched[{m}] unmatched[{x}]")
        if self.via_key:
            base += (f"; supported instead by {self.via_key} "
                     f"(score={self.via_score:.2f}), which {self.provision_key} "
                     f"expressly refers to")
        return base


class SupportScreen:
    """
    Corpus-calibrated topical support screen.

    tau_support is the score below which a claim-citation pair is flagged. It is
    calibrated empirically (see eval/run_support_eval.py) rather than assumed.
    """

    def __init__(self, db: MetadataDB, tau_support: float = 0.30,
                 min_terms: int = 3, graph=None):
        self.db = db
        self.tau = tau_support
        self.min_terms = min_terms
        # Optional statutory citation graph (retrieval/xref.py). Deterministic:
        # a dict of regex-extracted edges, no inference, so passing it here does
        # not weaken the guarantee. When absent, resolve() collapses to the
        # original two-valued behaviour.
        self.graph = graph
        self.idf = self._build_idf()

    def _build_idf(self) -> Dict[str, float]:
        provs = self.db.all_provisions()
        n = max(len(provs), 1)
        df = Counter()
        for p in provs:
            terms = set(self._terms(f"{p.marginal_note} {p.text or ''}"))
            df.update(terms)
        # Smoothed IDF; unseen terms get the maximum weight, since a claim term
        # absent from the entire corpus is maximally discriminative.
        return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _idf(self, term: str) -> float:
        n = max(self.db.count(), 1)
        return self.idf.get(term, math.log(n + 1) + 1.0)

    @staticmethod
    def _terms(text: str) -> List[str]:
        return [w for w in _WORD.findall((text or "").lower())
                if w not in _STRUCTURAL]

    def score(self, claim: str, provision_key: str) -> Optional[SupportScore]:
        prov = self.db.get(provision_key)
        if prov is None:
            return None

        # Remove citation spans so their tokens cannot inflate the score.
        stripped = claim
        for c in sorted(extract_citations(claim), key=lambda c: -c.start):
            stripped = stripped[:c.start] + " " + stripped[c.end:]

        claim_terms = set(self._terms(stripped))

        # MINIMUM-EVIDENCE GUARD.
        # A claim carrying little substantive vocabulary ("Charges were framed
        # under sections 318 and 103 of the BNS") gives the screen almost
        # nothing to work with: after structural terms are removed only
        # "charges" and "framed" remain, neither of which appears in the
        # provision, so the score collapses to zero and a perfectly valid claim
        # is flagged. The screen abstains below a minimum term count rather
        # than emitting a verdict it has no basis for. Abstention is recorded
        # as a score of 1.0 because V5 is advisory: declining to flag is the
        # conservative action.
        if len(claim_terms) < self.min_terms:
            return SupportScore(1.0, [], sorted(claim_terms), provision_key)

        prov_terms = set(self._terms(f"{prov.marginal_note} {prov.text or ''}"))

        matched = sorted(t for t in claim_terms if t in prov_terms)
        missed = sorted(t for t in claim_terms if t not in prov_terms)

        num = sum(self._idf(t) for t in matched)
        den = sum(self._idf(t) for t in claim_terms)
        return SupportScore(num / den if den else 0.0,
                            matched, missed, provision_key)

    def best_score(self, claim: str, keys: List[str]) -> Optional[SupportScore]:
        """Best support among several cited provisions in one sentence."""
        scored = [s for s in (self.score(claim, k) for k in keys) if s]
        return max(scored, key=lambda s: s.score) if scored else None

    # ------------------------------------------------------------------
    # Three-valued resolution, with support-via-reference
    # ------------------------------------------------------------------
    #
    # THE INTEGRATION GAP. Some provisions are not semantically self-contained:
    # their text merely refers elsewhere ("cognizance of any offence punishable
    # under section 138"). A claim about the SUBSTANCE of the offence, correctly
    # cited to such a provision, scores near zero against that provision's own
    # text and is flagged as unsupported -- precisely because it is one of the
    # provisions retrieval/xref.py exists to handle. The screen and the graph
    # were built independently and did not consult each other.
    #
    # WHY THIS IS NOT A BINARY RESCUE. The obvious repair -- suppress the flag
    # when a one-hop reference supports the claim -- was measured before being
    # implemented, and it costs flagging precision. The edge that rescues a
    # correct claim about a non-self-contained provision is the SAME edge that
    # rescues a topically-adjacent wrong one: a claim about when police may
    # arrest, wrongly cited to the procedure-after-arrest section, is "rescued"
    # through that section's reference to the arrest power. This is the
    # grammatical-role boundary already documented for V5, and a graph walk
    # cannot cross it either.
    #
    # We therefore do not suppress. Support-via-reference is reported as its own
    # weaker verdict, so a claim of this kind is never silently passed and never
    # mislabelled as unsupported. Both verdicts are advisory and neither strips.

    SUPPORTED = "supported"
    VIA_REFERENCE = "via_reference"
    WEAK = "weak"

    def resolve(self, claim: str, key: str):
        """
        Returns (verdict, SupportScore) for one claim-citation pair, where
        verdict is SUPPORTED, VIA_REFERENCE or WEAK. Returns (None, None) if
        the provision is absent from the metadata DB.
        """
        direct = self.score(claim, key)
        if direct is None:
            return None, None
        if direct.score >= self.tau:
            return self.SUPPORTED, direct

        if self.graph is not None:
            best = None
            for dst in sorted(self.graph.outbound(key)):
                s = self.score(claim, dst)
                if s is None or s.score < self.tau:
                    continue
                if best is None or s.score > best.score:
                    best = s
            if best is not None:
                direct.via_key = best.provision_key
                direct.via_score = best.score
                return self.VIA_REFERENCE, direct

        return self.WEAK, direct

    # Strongest verdict wins when a sentence cites several provisions.
    _RANK = {SUPPORTED: 2, VIA_REFERENCE: 1, WEAK: 0}

    def resolve_best(self, claim: str, keys: List[str]):
        """Best verdict among several provisions cited in one sentence."""
        out = [(v, s) for v, s in (self.resolve(claim, k) for k in keys) if v]
        if not out:
            return None, None
        return max(out, key=lambda t: (self._RANK[t[0]], t[1].score))
