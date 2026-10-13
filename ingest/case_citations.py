"""
Case-law citations: extraction from generated text, and resolution against
the judgment corpus (ingest/scrape_judgments.py).

WHY THIS EXISTS. Until this module the verifier checked statutory citations
only. A case name was never extracted, so "In Ramesh Kumar v. State of
Haryana (2019) 4 SCC 101 the Court held..." -- an invented precedent -- passed
verification untouched. That is exactly the failure Dahl et al. document for
US case law, and the paper's title and contribution C2 claim case law, so the
gap was a mismatch between the claim and the mechanism, not a missing
feature. A peer review (2026-10) said so directly.

WHAT IS CHECKED. Two citation forms, both deterministic:

  - party names, "A v. B" / "A vs B" / "A versus B": resolved when every
    distinctive word the answer gives for each side occurs in that side of a
    corpus judgment's title (Indian Kanoon titles, which may be truncated
    with "..."), with at least two distinctive words per side unless the
    title side itself has only one;
  - Supreme Court reporter citations -- SCC, SCR, SCALE, AIR (SC), SCC
    OnLine, AIROnline -- resolved by exact (reporter, year, volume, page)
    against the equivalent citations Indian Kanoon lists for each judgment.

A case citation that does not resolve is stripped, like a nonexistent
section: the guarantee is that every emitted citation resolves to the
corpus. With a 110-judgment corpus that means a REAL precedent outside the
corpus is stripped too -- the coverage cost of a closed-world check, which is
measured and reported rather than hidden (eval/run_case_citation_eval.py).

WHAT IS NOT CHECKED. That the case says what the answer claims it says
(the same resolution-vs-support distinction as for statutes), High Court
reporters, neutral citations, and case names written without "v."/"vs"
("the Arnesh Kumar judgment"). The fail-closed residual scan
(verify/residual.py) strips any other reporter-shaped citation it finds.
"""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Party names
# ---------------------------------------------------------------------------
_WORD = r"[A-Z][A-Za-z'’.&()\-]*"
_CONN = r"(?:of|and|the|for|in|at|&|@)"
_SIDE = rf"{_WORD}(?:\s+(?:{_CONN}\s+)*{_WORD}){{0,9}}"
CASE_NAME = re.compile(
    rf"(?P<p>{_SIDE})\s+(?:v\.?|vs\.?|versus)\s+(?P<r>{_SIDE})")

# Words that carry no identifying weight in a party name, or that introduce
# the name in prose ("In", "See", "Per").
_GENERIC = {
    "the", "of", "and", "in", "at", "for", "&", "@", "see", "per", "also",
    "cf", "following", "under", "as", "case", "judgment", "ltd", "limited",
    "pvt", "private", "co", "ors", "anr", "others", "another", "etc", "its",
    "thr", "through", "mr", "mrs", "ms", "dr", "smt", "shri", "sri", "sau",
    "m/s", "ms", "re", "court", "supreme", "high", "hon'ble", "honble",
    "landmark", "decision", "held", "where", "while", "when", "here",
}
_TOK = re.compile(r"[a-z0-9]+")


def _side_tokens(side: str) -> List[str]:
    return [t for t in _TOK.findall(side.lower().replace("'", ""))
            if t not in _GENERIC]


@dataclass(frozen=True)
class CaseName:
    petitioner: str
    respondent: str
    raw: str
    start: int
    end: int


def _is_act_name(side: str) -> bool:
    from ingest.normalize import ACT_ALIASES
    aliases = {a for al in ACT_ALIASES.values() for a in al}
    words = side.lower().strip().strip(".").split()
    return bool(words) and (" ".join(words) in aliases or words[-1] in aliases
                            or words[0] in aliases)


def extract_case_names(text: str) -> List[CaseName]:
    out = []
    for m in CASE_NAME.finditer(text):
        p = m.group("p")
        # "IPC vs BNS", "the CrPC v. the BNSS": two codes compared, not a case.
        if (_is_act_name(p.split()[-1]) and
                _is_act_name(m.group("r").split()[0])):
            continue
        # "The Supreme Court in Arnesh Kumar v. ..." -- the name starts after
        # the last connector "in"; earlier words are prose.
        cut = re.split(r"\s+in\s+", p)
        start = m.start("p") + (len(p) - len(cut[-1]))
        out.append(CaseName(petitioner=cut[-1], respondent=m.group("r"),
                            raw=text[start:m.end()], start=start, end=m.end()))
    return out


# ---------------------------------------------------------------------------
# Supreme Court reporter citations
# ---------------------------------------------------------------------------
_Y = r"(?P<year>(?:18|19|20)\d\d)"
_REPORTERS = [
    # (2018) 6 SCC 454   (2019) 11 SCALE 209   [1978] 2 SCR 621
    re.compile(rf"[\(\[]\s*{_Y}\s*[\)\]]\s*(?P<vol>\d{{1,3}})\s+"
               rf"(?P<rep>S\.?\s?C\.?\s?C\.?|S\.?\s?C\.?\s?R\.?|SCALE)"
               rf"(?:\s*\(\s*Cri\s*\))?\s+(?P<page>\d{{1,5}})", re.I),
    # 2018 (6) SCC 454
    re.compile(rf"\b{_Y}\s*\(\s*(?P<vol>\d{{1,3}})\s*\)\s*"
               rf"(?P<rep>SCC|SCR|SCALE)(?:\s*\(\s*Cri\s*\))?\s+(?P<page>\d{{1,5}})", re.I),
    # AIR 2018 SC 1498 / AIR 2018 SUPREME COURT 1498
    re.compile(rf"\b(?P<rep>AIR)\s*{_Y}\s*(?:SC|S\.C\.|SUPREME\s+COURT)\s+"
               rf"(?P<page>\d{{1,5}})", re.I),
    # 2019 SCC OnLine SC 1234
    re.compile(rf"\b{_Y}\s+(?P<rep>SCC\s*On\s*Line)\s+SC\s+(?P<page>\d{{1,6}})", re.I),
    # AIRONLINE 2019 SC 2430
    re.compile(rf"\b(?P<rep>AIR\s*ONLINE)\s+{_Y}\s+SC\s+(?P<page>\d{{1,6}})", re.I),
]


def _rep_code(rep: str) -> str:
    r = re.sub(r"[^A-Z]", "", rep.upper())
    return {"SCC": "SCC", "SCR": "SCR", "SCALE": "SCALE", "AIR": "AIR",
            "SCCONLINE": "SCCOL", "AIRONLINE": "AIROL"}.get(r, r)


@dataclass(frozen=True)
class ReporterCitation:
    key: str            # "SCC|2018|6|454", "AIR|2018||1498"
    raw: str
    start: int
    end: int


def extract_reporter_citations(text: str) -> List[ReporterCitation]:
    out, claimed = [], []
    for pat in _REPORTERS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(not (e <= cs or s >= ce) for cs, ce in claimed):
                continue
            gd = m.groupdict()
            key = "|".join([_rep_code(gd["rep"]), gd["year"],
                            (gd.get("vol") or "").lstrip("0"),
                            gd["page"].lstrip("0")])
            out.append(ReporterCitation(key=key, raw=text[s:e], start=s, end=e))
            claimed.append((s, e))
    out.sort(key=lambda r: r.start)
    return out


# ---------------------------------------------------------------------------
# The index the verifier resolves against
# ---------------------------------------------------------------------------
@dataclass
class _Entry:
    doc_id: str
    case_name: str
    pet: List[str]
    res: List[str]
    pet_truncated: bool
    res_truncated: bool


def _side_matches(answer: List[str], title: List[str], truncated: bool) -> bool:
    if not answer or not title:
        return False
    need = min(2, len(set(title)))
    a, t = set(answer), set(title)
    if a <= t and len(a) >= need:
        return True
    # A truncated title ("Communist Party Of India (Marxist) Its ...") shows
    # only a prefix of the party name: accept when every word it does show
    # is in the answer's name.
    return truncated and t <= a and len(t) >= need


class CaseIndex:
    """Deterministic lookup of case names and reporter citations."""

    def __init__(self, judgments: Iterable = ()):
        self.entries: List[_Entry] = []
        self.by_reporter: Dict[str, str] = {}
        self.names: Dict[str, str] = {}
        for j in judgments:
            parts = re.split(r"\s+vs\.?\s+", j.case_name, maxsplit=1)
            pet, res = (parts + [""])[:2]
            self.entries.append(_Entry(
                doc_id=j.doc_id, case_name=j.case_name,
                pet=_side_tokens(pet), res=_side_tokens(res),
                pet_truncated=pet.rstrip().endswith("..."),
                res_truncated=res.rstrip().endswith("...")))
            self.names[j.doc_id] = j.case_name
            for c in j.citations:
                for rc in extract_reporter_citations(c):
                    self.by_reporter.setdefault(rc.key, j.doc_id)

    def __len__(self):
        return len(self.entries)

    def resolve_name(self, name: CaseName) -> Optional[str]:
        pet, res = _side_tokens(name.petitioner), _side_tokens(name.respondent)
        # The answer's respondent may run on into prose ("State of Bihar
        # Court held"); try successively shorter prefixes of it. Only the
        # END is trimmed -- trimming words off the front of the petitioner
        # would let "Fake Kumar" match "Arnesh Kumar".
        for e in self.entries:
            if not _side_matches(pet, e.pet, e.pet_truncated):
                continue
            for n in range(len(res), 0, -1):
                if _side_matches(res[:n], e.res, e.res_truncated):
                    return e.doc_id
        return None

    def resolve_reporter(self, cit: ReporterCitation) -> Optional[str]:
        return self.by_reporter.get(cit.key)


class AttestedCases:
    """Case citations that occur in the TEXT of retrieved sources.

    The statutory corpus itself cites cases: Indian Kanoon's IT Act s.66A
    reads "Held unconstitutional by the Supreme Court in Shreya Singhal v
    Union of India, AIR 2015 SC 1523". An answer repeating that citation is
    not fabricating it -- the source it was given attests it -- so a case
    citation also resolves when it occurs in a retrieved chunk. Still closed
    world and still deterministic: the citation must be in the corpus text the
    answer was grounded on."""

    def __init__(self, texts: Iterable[str]):
        self.names = []
        self.reporters: Set[str] = set()
        for t in texts:
            for n in extract_case_names(t):
                self.names.append((_side_tokens(n.petitioner),
                                   _side_tokens(n.respondent)))
            for r in extract_reporter_citations(t):
                self.reporters.add(r.key)

    def has_name(self, name: CaseName) -> bool:
        pet, res = _side_tokens(name.petitioner), _side_tokens(name.respondent)
        for tp, tr in self.names:
            if _side_matches(pet, tp, False) and any(
                    _side_matches(res[:n], tr, False)
                    for n in range(len(res), 0, -1)):
                return True
        return False

    def has_reporter(self, cit: ReporterCitation) -> bool:
        return cit.key in self.reporters


def build_case_index() -> CaseIndex:
    from ingest.scrape_judgments import scrape_all
    return CaseIndex(scrape_all())
