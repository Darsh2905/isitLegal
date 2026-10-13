"""
Fail-closed residual scan: the backstop behind the citation parser.

THE PROBLEM IT CLOSES. Every check in verifier.py runs on what the parser
extracts. A citation the parser does not recognise is not checked -- and
until this module, not checked meant PASSED. Twelve parser defects (D1-D12)
were found that way, each one an evasion that left a fabricated citation in
the emitted answer, and a 2026-10 audit found more in minutes (a bare
"Section 7777" with no act name, "IPC's section 7777", "Section 7777 (IPC)",
"Section seven thousand", "the 7777th section", "Clause (9) of Article 22",
every case name). A peer review put it plainly: a parser-based guarantee is
fail-open, and after twelve evasion defects a fail-closed design is owed.

WHAT IT DOES. Independently of the parser, it finds every span that LOOKS
like a legal reference -- a provision word followed by a number or a
parenthesised label, a section symbol, a known act name next to a number, an
ordinal section, a "v."/"vs" case name, a reporter citation -- and checks
that the reference's number (its anchor) lies inside a span some extractor
claimed. A reference-shaped span no extractor claimed is one the verifier
cannot have checked, so the sentence holding it is stripped.

The scan is deliberately broader than the parser. A broad scan costs
coverage when it fires on something harmless; a narrow one reopens the hole.
Its trigger rate on real generated output is measured and reported
(eval/run_residual_eval.py) rather than assumed small, and every instance is
listed so the cost can be read rather than inferred.

WHAT IT DOES NOT COVER. Group references with no provision number -- "Part
III of the Constitution", "Chapter XVI", "the Seventh Schedule" -- are not
provision citations and are left alone; so are bare act names ("the BNSS
applies"), which cite no provision and so cannot cite a nonexistent one.

No model, no inference: regular expressions over the answer text only
(CLAUDE.md invariant 1).
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from ingest.normalize import ACT_ALIASES, _ACT_YEARS, _sanitise

_NUMWORD = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
            r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
            r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
            r"|thousand)")
_ALIASES = sorted({a for al in ACT_ALIASES.values() for a in al}, key=len,
                  reverse=True)
_ACT = "|".join(re.escape(a) for a in _ALIASES)
_ALIAS_CODE = {a: c for c, al in ACT_ALIASES.items() for a in al}

# Each signal names the group ("n") whose start is the reference's anchor.
_SIGNALS = [
    ("provision word + number",
     re.compile(r"(?<!['’])\b(?:sections?|secs?|ss?|u/s|articles?|arts?"
                r"|sub-?sections?|clauses?|sub-?clauses?|provisos?|rules?"
                r"|paragraphs?|paras?)\b\.?[^\S\n]*(?:no\.?[^\S\n]*)?"
                rf"(?P<n>\d|{_NUMWORD}\b|\((?!s\))\s*[0-9a-z]{{1,5}}\s*\))",
                re.I)),
    ("section symbol", re.compile(r"(?P<n>§+)[^\S\n]*\(?[^\S\n]*\d")),
    ("act name + number",
     re.compile(rf"\b(?P<act>{_ACT})(?!\w)(?:['’]s)?[\s,:(]{{0,3}}"
                rf"(?:no\.?\s*)?(?P<n>\d{{1,4}}[A-Z]?)\b", re.I)),
    ("number + act name",
     re.compile(rf"(?<![\d.])(?P<n>\d{{1,4}}[A-Z]?)[^\S\n]*[,(]?[^\S\n]*"
                rf"(?:of[^\S\n]+(?:the[^\S\n]+)?)?"
                rf"\(?(?P<act>{_ACT})(?!\w)", re.I)),
    ("ordinal provision",
     re.compile(r"\b(?P<n>\d+)(?:st|nd|rd|th)\s+(?:section|article|clause"
                r"|sub-?section)\b", re.I)),
    ("case name",
     re.compile(r"\b[A-Z][\w.'’&\-]*\s+(?P<n>v\.?|vs\.?|versus)\s+[A-Z]")),
    # A lenticular marker that is not "\u3010n\u3011" -- gpt-oss-120b's native
    # "\u3010\u2020L1-L3\u3011" style -- names no retrieved chunk V1 can check.
    ("unreadable marker",
     re.compile(r"(?P<n>\u3010(?!\s*\d+\s*\u3011)[^\u3011]{0,40}\u3011)")),
    ("reporter citation",
     re.compile(r"(?P<n>[\(\[]\s*(?:18|19|20)\d\d\s*[\)\]]\s*\d{1,3}\s+[A-Z][A-Za-z.]+"
                r"|\b(?:18|19|20)\d\d\s*\(\s*\d{1,3}\s*\)\s*[A-Z]{2,}"
                r"|\bAIR\s*(?:18|19|20)\d\d"
                r"|\bSCC\s+On\s*Line|\bAIR\s*ONLINE\b)")),
]


@dataclass(frozen=True)
class Residual:
    kind: str
    raw: str
    start: int
    end: int


# Words after which a number is part of a provision, or a group of them, not a
# provision of its own: "Exception 1 of the IPC", "Chapter 20 of the BNSS".
_PART_WORDS = re.compile(r"(?i)\b(?:exceptions?|explanations?|illustrations?"
                         r"|chapters?|parts?|schedules?|orders?|forms?|items?"
                         r"|entr(?:y|ies)|lists?)\s*$")


def _not_a_provision(scan: str, m) -> bool:
    """A number beside an act name that is not a provision reference: a year
    ("1 July 2024 (BNS)"), an act number ("Indian Penal Code (45 of 1860)",
    which Indian Kanoon inlines in the statutory text itself), or a part of
    a provision ("Exception 1 of the IPC")."""
    if "act" not in m.groupdict() or m.group("act") is None:
        return False
    n = re.match(r"\d+", m.group("n") or "")
    if n and len(n.group()) == 4 and n.group()[:2] in ("18", "19", "20"):
        return True
    if re.match(r"\s*of\s+(?:18|19|20)\d\d", scan[m.end("n"):m.end("n") + 12]):
        return True
    return bool(_PART_WORDS.search(scan[max(0, m.start("n") - 15):m.start("n")]))


def _is_enactment_year(m) -> bool:
    """"BNS 2023", "IPC, 1860": an act followed (or preceded) by its own
    year is a name, not a provision reference."""
    act = m.groupdict().get("act")
    if not act:
        return False
    code = _ALIAS_CODE.get(act.lower())
    return m.group("n") in _ACT_YEARS.get(code, set())


def _act_comparison(scan: str, m) -> bool:
    """"IPC vs BNS", "the CrPC v. the BNSS": a comparison of two codes, not a
    case name."""
    before = re.findall(r"[\w.]+", scan[max(0, m.start() - 40):m.start("n")])
    after = re.findall(r"[\w.]+", scan[m.end("n"):m.end("n") + 40])
    def is_act(w):
        return w.lower().strip(".") in _ALIAS_CODE or w.lower() in _ALIAS_CODE
    return bool(before and after and is_act(before[-1]) and is_act(after[0]))


def residual_references(text: str,
                        covered: Iterable[Tuple[int, int]]) -> List[Residual]:
    """Reference-shaped spans whose anchor lies in no covered span."""
    scan = _sanitise(text)
    spans = list(covered)
    out, seen = [], set()
    for kind, pat in _SIGNALS:
        for m in pat.finditer(scan):
            a = m.start("n")
            if any(s <= a < e for s, e in spans):
                continue
            if _is_enactment_year(m) or _not_a_provision(scan, m):
                continue
            if kind == "case name" and _act_comparison(scan, m):
                continue
            if (m.start(), m.end()) in seen:
                continue
            seen.add((m.start(), m.end()))
            out.append(Residual(kind=kind, raw=text[m.start():m.end()],
                                start=m.start(), end=m.end()))
    out.sort(key=lambda r: r.start)
    return out
