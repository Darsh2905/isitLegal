"""
Citation normalisation for Indian statutory references.

Maps the many surface forms a legal citation takes in natural language onto a
single canonical key, e.g.

    "Sec. 420 IPC"                        -> IPC§420
    "s.420 of the Indian Penal Code"      -> IPC§420
    "section 420, IPC"                    -> IPC§420
    "Section 138 of the NI Act"           -> NIA§138
    "Article 21"                          -> CONST§21
    "u/s 302 IPC"                         -> IPC§302
    "BNS Section 103(1)"                  -> BNS§103(1)

This module is a precondition for the citation verifier: a generated citation
cannot be checked against the corpus unless both are expressed in a common
referential vocabulary.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List

# ---------------------------------------------------------------------------
# Act vocabulary.  Each entry: canonical code -> list of surface aliases.
# Aliases are matched case-insensitively, longest-first.
# ---------------------------------------------------------------------------
ACT_ALIASES = {
    "IPC": [
        "indian penal code, 1860", "indian penal code 1860", "indian penal code",
        "i.p.c.", "ipc", "penal code",
    ],
    "BNS": [
        "bharatiya nyaya sanhita, 2023", "bharatiya nyaya sanhita 2023",
        "bharatiya nyaya sanhita", "b.n.s.", "bns", "nyaya sanhita",
    ],
    "CRPC": [
        "code of criminal procedure, 1973", "code of criminal procedure 1973",
        "code of criminal procedure", "criminal procedure code",
        "cr.p.c.", "crpc", "cr p c",
    ],
    "BNSS": [
        "bharatiya nagarik suraksha sanhita, 2023", "bharatiya nagarik suraksha sanhita",
        "b.n.s.s.", "bnss", "nagarik suraksha sanhita",
    ],
    "IEA": [
        "indian evidence act, 1872", "indian evidence act 1872",
        "indian evidence act", "evidence act", "i.e.a.", "iea",
    ],
    "BSA": [
        "bharatiya sakshya adhiniyam, 2023", "bharatiya sakshya adhiniyam",
        "b.s.a.", "bsa", "sakshya adhiniyam",
    ],
    "NIA": [
        "negotiable instruments act, 1881", "negotiable instruments act",
        "n.i. act", "ni act", "nia",
    ],
    "ITA": [
        "information technology act, 2000", "information technology act",
        "i.t. act", "it act", "ita",
    ],
    "CONST": [
        "constitution of india", "indian constitution", "the constitution",
        "constitution",
        # "Const §22" -- written by gpt-oss-20b in the live round; without the
        # alias the bare "§22" was placed in whatever act was cited earlier.
        "const.", "const",
    ],
    "ICA": [
        "indian contract act, 1872", "indian contract act", "contract act",
    ],
}

# Build a single alternation, longest alias first so "indian penal code"
# is preferred over the bare "ipc" substring inside it.
_ALL_ALIASES = sorted(
    ((alias, code) for code, aliases in ACT_ALIASES.items() for alias in aliases),
    key=lambda t: -len(t[0]),
)
_ALIAS_TO_CODE = dict(_ALL_ALIASES)
_ACT_PATTERN = "|".join(re.escape(a) for a, _ in _ALL_ALIASES)

# Section-number token: 302, 420A, 103(1), 66-A, 376(2)(n)
# The letter suffix must be immediately adjacent to the digits and must not be
# followed by another letter -- otherwise "Article 14 of the Constitution"
# captures the "o" of "of" as a suffix, and "154 Cr.P.C." captures the "C".
# The hyphenated suffix ("65-B", "66-A") is at most three characters and never
# crosses a line break: "BSA \u00a76**\n- Declares that..." -- a heading then a
# bullet -- otherwise parsed as section "6-Declares".
_NUM = (
    r"\d+[A-Za-z]?(?![A-Za-z])"
    r"(?:[^\S\n]*[-–][^\S\n]*[A-Za-z0-9]{1,3}(?![A-Za-z0-9]))?"
    r"(?:\s*\([0-9a-zA-Z]+\))*"
)

# Words that introduce a section reference
# "s 420" (no period) is a section word only when a number follows; "§"/"§§"
# are matched separately because a word boundary cannot precede them.
_SEC_WORD = (r"(?:(?:(?<!['\u2019])\b(?:sections?|sec\.?|ss\.?|s\.|s(?=\s*\d)|u/s"
             r"|under\s+section))|§§?)")
# P3 (act first) must not treat "§" as its section word: "the Bharatiya Nyaya
# Sanhita §318, §7777" is a list for P4B, and P3 claiming "§318" alone left
# "§7777" to the weaker bare-reference path.
_SEC_WORD_NO_SYMBOL = (r"(?:(?<!['\u2019])\b(?:sections?|sec\.?|ss\.?|s\.|s(?=\s*\d)"
                       r"|u/s|under\s+section))")
# A footnote or citation marker may sit between the section number and the act
# name ("Section 420[1] IPC"), which severed the two and defeated P1/P2.
_FOOTNOTE = r"(?:\s*\[\d+\])?"
_ART_WORD = r"(?:articles?|art\.?)"

# A single section number with no range part: safe to repeat inside a list,
# where the range separator would otherwise be ambiguous with a list separator.
_NUM_ATOM = r"\d+[A-Za-z]?(?![A-Za-z])(?:\s*\([0-9a-zA-Z]+\))*"

# Separators that join members of an enumeration. Slash ("302/304"), semicolon,
# and the "r/w" / "read with" conjunction are standard in Indian legal drafting
# and were all absent, so a fabricated section joined by any of them escaped
# extraction entirely.
_SEP = (r"(?:\s*,\s*|\s*;\s*|\s+and\s+|\s+or\s+|\s*&\s*|\s+to\s+|\s*/\s*"
        r"|\s+r/w\s+|\s+rw\s+|\s+read\s+with\s+|\s*[-\u2013]\s*)")


# Characters that carry no citation meaning yet sit inside a citation and defeat
# every pattern below. Adversarial audit found six evasion constructs built from
# them: markdown emphasis around the section number ("Section **420** IPC"),
# which generators emit constantly, and zero-width characters, which hide a
# fabricated section from the existence check entirely.
#
# They are replaced with SPACES OF EQUAL LENGTH rather than deleted, because
# Citation.start/end are character offsets into the caller's text: repair rule
# R2 splices a rewritten reference in by offset and the support screen excises
# citation spans by offset, so a shifted offset would corrupt emitted output.
_NOISE = re.compile(r"[*_`]+|[\u200b\u200c\u200d\u2060\ufeff\u00ad]+")

# Hyphen-lookalikes a generator emits interchangeably with a plain ASCII "-"
# inside a compound section number ("113-B", "66-A"): non-breaking hyphen,
# figure dash, en dash, em dash, minus sign. Found via a live Tab 6 answer
# that wrote "\u00a7113\u2011B" (non-breaking hyphen) -- none of the patterns below
# recognise anything but "-"/"\u2013", so the unrecognised lookalike didn't just
# fail to match the suffix, it truncated the citation to its bare digits and
# let the shorter number silently resolve against IEA\u00a7113 ("Proof of cession
# of territory"), a real but wholly unrelated provision -- a false VALIDATION,
# not a blind spot. Normalised to ASCII "-" (one character for one, so every
# offset stays valid) rather than blanked, since blanking would still break
# the compound-suffix match.
_HYPHENS = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")


def _sanitise(text: str) -> str:
    """Blank citation-irrelevant noise and normalise hyphen lookalikes,
    preserving every character offset."""
    text = _HYPHENS.sub("-", text)
    return _NOISE.sub(lambda m: " " * len(m.group(0)), text)


@dataclass(frozen=True)
class Citation:
    """A normalised legal reference."""
    act: str          # canonical act code, e.g. "IPC"
    section: str      # normalised section token, e.g. "420" or "103(1)"
    raw: str          # the original surface string as it appeared
    start: int        # char offset in source text
    end: int

    @property
    def key(self) -> str:
        return f"{self.act}§{self.section}"

    def __str__(self) -> str:
        return self.key


def _clean_section(tok: str) -> str:
    """Normalise whitespace and dashes inside a section token."""
    tok = tok.strip().rstrip(".,;:")
    # A list element carried over from P4B's per-item [§#] allowance keeps
    # its own leading symbol after list-splitting ("§118", not "118") --
    # strip it here so every caller gets a plain section token regardless of
    # which pattern produced it. Harmless elsewhere: no other caller's token
    # ever starts with §/# to begin with.
    tok = tok.lstrip("§# \t")
    tok = re.sub(r"\s+", "", tok)
    tok = tok.replace("–", "-")
    # Uppercase a trailing letter suffix: 66a -> 66A, 420b -> 420B
    tok = re.sub(r"(\d)([a-z])\b", lambda m: m.group(1) + m.group(2).upper(), tok)
    # Pre-existing bug, found while investigating the two above: a hyphenated
    # compound suffix ("65-B", "66-A") kept its hyphen instead of dropping
    # it, but every key in the real 3,263-provision corpus joins a suffix
    # letter directly with no separator ("IEA§65B", never "IEA§65-B") --
    # checked against every hyphenated key in the corpus (zero exist). Since
    # "Section 65-B" is a very common real-world citation style (it's this
    # project's own paper's worked example -- the Evidence Act's
    # electronic-records provision), this silently orphaned an entire
    # citation style: it either matched a wrong, unrelated shorter
    # provision (before the P4/_HYPHENS fixes above) or, once those made the
    # suffix reach this point at all, would have still failed to resolve
    # here against the real key. Strip the hyphen rather than keep it.
    tok = re.sub(r"-([a-zA-Z])\b", lambda m: m.group(1).upper(), tok)
    return tok


def _resolve_act(surface: str) -> Optional[str]:
    if surface is None:
        return None
    return _ALIAS_TO_CODE.get(surface.lower().strip())


# --- Patterns, tried in order -------------------------------------------------
# P1: "Section 420 of the Indian Penal Code" / "s.420 of the IPC"
_P1 = re.compile(
    rf"{_SEC_WORD}\s*\(?\s*(?P<num>{_NUM})\s*\)?{_FOOTNOTE}"
    rf"\s*(?:,)?\s*of\s+(?:the\s+)?(?P<act>{_ACT_PATTERN})(?!\w)",
    re.IGNORECASE,
)
# P2: "Section 420 IPC" / "Section 420, IPC" / "u/s 302 IPC"
_P2 = re.compile(
    rf"{_SEC_WORD}\s*\(?\s*(?P<num>{_NUM})\s*\)?{_FOOTNOTE}"
    rf"\s*(?:,|\s)\s*\(?\s*(?P<act>{_ACT_PATTERN})(?!\w)",
    re.IGNORECASE,
)
# P3: "IPC Section 420" / "BNS s. 103" / "the Indian Penal Code (Section 500)"
# The trailing "(?)" pair is optional and matches on either side independently
# (a literal "(" is not whitespace or a comma, so it previously broke the
# separator entirely): found live, not by adversarial construction --
# "punishable under the Indian Penal Code (Section 500)" extracted nothing,
# while the identical claim with a comma or "of the" in place of the
# parenthesis extracted correctly. A generator naming the act in prose and
# then parenthesising the pinpoint section is at least as natural a
# construct as any of D1-D11, and was invisible to V2 in exactly the same
# way: not a partial match, zero citations from the whole clause.
# An enactment year may sit between the act and the section word ("IT Act 2000
# Section 66", "IPC, 1860, section 302") -- without the optional year group
# the section went unextracted entirely once P4 stopped misreading the year
# itself as the section number.
_P3 = re.compile(
    rf"\b(?P<act>{_ACT_PATTERN})(?:['\u2019]s)?\s*(?:,)?\s*(?:(?:18|19|20)\d{{2}}\s*,?\s*)?"
    rf"\(?\s*{_SEC_WORD_NO_SYMBOL}\s*(?P<num>{_NUM})\s*\)?",
    re.IGNORECASE,
)
# P4: "IPC 420" / "IPC §420" / "BNS §103(1)"
# The num group used to be a bespoke inline pattern rather than the shared
# _NUM token every other single-citation pattern (P1/P2/P3/P5) uses, and it
# never grew _NUM's hyphenated-compound-suffix branch. "Section 65-B" -- an
# extremely common real-world citation style for the Evidence Act's
# electronic-records provision, hyphen and all -- silently truncated to
# "65", which happens to be a real but unrelated provision ("secondary
# evidence"), so the wrong citation validated cleanly instead of being
# caught. Found live, on the exact provision this project's own paper uses
# as its worked example. _NUM is list-unsafe (its hyphen branch would eat a
# range separator), but P4 matches a single bare citation, not a list, so
# that risk doesn't apply here -- unlike P4B, which deliberately uses the
# list-safe _NUM_ATOM instead.
_P4 = re.compile(
    rf"\b(?P<act>{_ACT_PATTERN})\s*[§#]?\s*(?P<num>{_NUM})\b",
    re.IGNORECASE,
)
# P4B: act-first BARE enumeration with no section word -- "IPC 302/304",
# "BNS 103, 105". P4 handles the single-number form; without this, every member
# after the first was dropped.
# The list element must carry the same negative lookahead as _NUM. Without it
# the optional letter suffix eats the leading letter of the following word --
# "105 and 7777" parses as element "105 a", the separator then fails to match,
# and the list silently truncates to its first two members. This is the same
# greedy-suffix root cause as the two defects reported in Section VII-B.
# A generator that writes a bare list under one act name often repeats the
# section symbol per item rather than dropping it after the first --
# "the Bharatiya Nyaya Sanhita §80, §118" -- which the separator alone
# doesn't expect. Without the optional [§#] before each continuation atom,
# the "+" repetition group can't consume ", §118" at all (a literal "§"
# sits where a digit must start the next atom), so the whole group matches
# zero times and P4B doesn't fire on the list beyond its first member --
# not a partial match, a silently dropped one, found via a live Tab 6 answer
# where BNS§118 vanished with no finding raised for it at all.
_P4B = re.compile(
    rf"\b(?P<act>{_ACT_PATTERN})\s*[§#]?\s*"
    rf"(?P<list>{_NUM_ATOM}(?:{_SEP}[§#]?\s*{_NUM_ATOM})+)",
    re.IGNORECASE,
)

# P5: "Article 21" / "Art. 19(1)(a)"  -> always the Constitution
_P5 = re.compile(rf"\b{_ART_WORD}\s*(?P<num>{_NUM})", re.IGNORECASE)

# P0: ENUMERATED references -- "sections 302 and 304 of the IPC",
# "Ss. 420, 468 and 471 IPC", "sections 302 to 304 IPC".
#
# These previously matched NOTHING, not merely partially: the trailing "of the
# ACT" is separated from the first number by the rest of the list, so P1 fails,
# and P2 requires the act immediately after a single number. A fabricated
# section concealed inside an enumeration was therefore invisible to the
# verifier's existence check -- a hole in the guarantee rather than a gap in
# coverage. P0 is tried first so it claims the whole span.
_P0 = re.compile(
    rf"(?:\b(?:sections?|secs?\.?|ss\.?|arts?\.?|articles?)|§§?)\s*"
    rf"(?P<list>{_NUM}(?:{_SEP}{_NUM})+)"
    rf"\s*(?:,)?\s*\(?\s*(?:of\s+(?:the\s+)?)?(?P<act>{_ACT_PATTERN})(?!\w)",
    re.IGNORECASE,
)
# P0B: ACT-FIRST enumeration -- "IPC sections 302 and 304", "BNS ss. 103, 105".
# The mirror image of the P0 defect and equally total: P3 matches the act plus
# the FIRST number only, so every later member of the list -- including a
# fabricated one -- was silently dropped rather than partially reported.
_P0B = re.compile(
    rf"\b(?P<act>{_ACT_PATTERN})(?:['\u2019]s)?\s*(?:,)?\s*"
    rf"(?:\b(?:sections?|secs?\.?|ss\.?|arts?\.?|articles?)|§§?)\s*"
    rf"(?P<list>{_NUM}(?:{_SEP}{_NUM})+)",
    re.IGNORECASE,
)

# Same, but Constitution articles carry no trailing act name.
_P0_ART = re.compile(
    rf"\b(?:articles|arts?\.)\s*(?P<list>{_NUM}(?:{_SEP}{_NUM})+)",
    re.IGNORECASE,
)
_RANGE_SEP = re.compile(r"\s+to\s+|\s*[-\u2013]\s*", re.IGNORECASE)
_LIST_SPLIT = re.compile(
    r"\s*,\s*|\s*;\s*|\s+and\s+|\s+or\s+|\s*&\s*|\s*/\s*"
    r"|\s+r/w\s+|\s+rw\s+|\s+read\s+with\s+", re.IGNORECASE)


def _expand_list(raw: str):
    """
    Expand an enumeration into individual section tokens.

    Ranges ("302 to 304") are expanded only when both endpoints are plain
    integers and the span is small; a range over lettered or sub-sectioned
    tokens cannot be enumerated safely, so the endpoints alone are returned.
    Over-expanding would invent provisions, which is the failure the verifier
    exists to prevent.
    """
    out = []
    for part in _LIST_SPLIT.split(raw):
        part = part.strip()
        if not part:
            continue
        if _RANGE_SEP.search(part):
            ends = [e.strip() for e in _RANGE_SEP.split(part) if e.strip()]
            if len(ends) == 2 and all(e.isdigit() for e in ends):
                a, b = int(ends[0]), int(ends[1])
                if 0 <= b - a <= 50:
                    out.extend(str(n) for n in range(a, b + 1))
                    continue
            out.extend(ends)
        else:
            out.append(part)
    return out


# P6: a number directly followed by an act, with no section word -- "charged
# under 302 IPC", and the second member of a mixed-act list, "Sections 198
# CrPC and 219 BNSS" (found in live gpt-oss-120b output: the residual scan
# stripped it, correctly but at the cost of a valid citation). Section numbers
# in these Acts are below 1000, so the number is limited to three digits:
# a year beside an act name is never read as a section by this pattern.
_P6 = re.compile(
    rf"(?<![\w.§/\-])(?P<num>\d{{1,3}}[A-Z]{{0,2}}(?![\w])(?:\s*\([0-9a-zA-Z]+\))*)"
    rf"\s+(?P<act>{_ACT_PATTERN})(?!\w)",
    re.IGNORECASE,
)

_PATTERNS = [_P0, _P0B, _P1, _P2, _P3, _P4B, _P4, _P6]
_ACT_BEFORE = re.compile(rf"(?:{_ACT_PATTERN})['\u2019]?s?[\s*_`]*$", re.IGNORECASE)

# "IPC 1860", "BNS 2023", "the IT Act 2000": an act name followed by its own
# enactment year is not a section reference, but the bare act-first patterns
# (P4/P4B) read it as one -- "The IPC 1860 was replaced by the BNS 2023" became
# IPC§1860 and BNS§2023, both nonexistent, so V2 stripped a correct sentence.
# Only an act's OWN enactment (or principal amendment) year is read as a year
# rather than a section: an earlier version treated any 1800-2099 number that
# way, which let a fabricated "IPC 1999 punishes cyber stalking" through V2
# unchecked. An explicit "IPC §1999" is always extracted as a section.
_ACT_YEARS = {
    "IPC": {"1860"}, "BNS": {"2023"}, "BNSS": {"2023"}, "BSA": {"2023"},
    "CRPC": {"1973", "1974"}, "IEA": {"1872"}, "ICA": {"1872"},
    "NIA": {"1881"}, "ITA": {"2000", "2008"}, "CONST": {"1949", "1950"},
}


def _is_year_not_section(scan: str, m, pat) -> bool:
    if pat is _P4:
        num_start, first = m.start("num"), m.group("num")
    elif pat is _P4B:
        num_start = m.start("list")
        first = _LIST_SPLIT.split(m.group("list"))[0]
    else:
        return False
    between = scan[m.end("act"):num_start]
    act = _resolve_act(m.group("act"))
    return ("§" not in between and "#" not in between
            and first.strip() in _ACT_YEARS.get(act, set()))


def extract_citations(text: str) -> List[Citation]:
    # Pure in `text`, and called several times per sentence (the verifier,
    # the unresolved-act and bare-reference extractors, both repair rules):
    # memoised so each sentence is parsed once.
    return list(_extract_citations_cached(text))


@lru_cache(maxsize=8192)
def _extract_citations_cached(text: str):
    return tuple(_extract_citations(text))


def _extract_citations(text: str) -> List[Citation]:
    """
    Extract and normalise every statutory reference in `text`.

    Returns citations in order of appearance. Overlapping matches are resolved
    by preferring the earliest, then the longest match (so "Section 420 of the
    Indian Penal Code" is not also matched as the bare "Indian Penal Code").
    """
    found: List[Citation] = []
    claimed: List[tuple] = []  # (start, end) spans already consumed

    # Match against the sanitised view, but report offsets and raw spans from
    # the caller's original text: the sanitiser is length-preserving, so an
    # offset into one is an offset into the other.
    scan = _sanitise(text)

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in claimed)

    # Ordered patterns: more specific forms first.
    for pat in _PATTERNS:
        for m in pat.finditer(scan):
            s, e = m.span()
            # A span never ends in whitespace: P3's optional closing
            # parenthesis let "IPC Section 302 " carry its trailing space, and
            # repair rule R2, splicing by offset, then glued the next word to
            # its own closing parenthesis ("(formerly IPC section 302
            # )punishes"). Found by the fail-closed probe set's corruption check.
            e = s + len(scan[s:e].rstrip())
            if overlaps(s, e):
                continue
            # A section symbol belongs to the act written BEFORE it: in
            # "BNSS \u00a7 401, CRPC \u00a7 360", P1/P2 (section word, then act)
            # read "\u00a7 401, CRPC" as CrPC 401, and repair R2 then rewrote
            # that misreading into the answer (found in the support labelling
            # of the second live round). When an act name immediately precedes
            # the symbol, the number is that act's.
            if (pat in (_P1, _P2) and scan[s:e].lstrip().startswith("§")
                    and _ACT_BEFORE.search(scan[max(0, s - 60):s])):
                continue
            act = _resolve_act(m.group("act"))
            if act is None:
                continue
            if _is_year_not_section(scan, m, pat):
                continue
            gd = m.groupdict()
            if gd.get("list"):
                for num in _expand_list(gd["list"]):
                    found.append(Citation(act=act, section=_clean_section(num),
                                          raw=text[s:e], start=s, end=e))
            else:
                found.append(
                    Citation(act=act, section=_clean_section(m.group("num")),
                             raw=text[s:e], start=s, end=e)
                )
            claimed.append((s, e))

    # Enumerated articles -> Constitution
    for m in _P0_ART.finditer(scan):
        s, e = m.span()
        if overlaps(s, e):
            continue
        for num in _expand_list(m.group("list")):
            found.append(Citation(act="CONST", section=_clean_section(num),
                                  raw=text[s:e], start=s, end=e))
        claimed.append((s, e))

    # Articles -> Constitution
    for m in _P5.finditer(scan):
        s, e = m.span()
        if overlaps(s, e):
            continue
        found.append(
            Citation(act="CONST", section=_clean_section(m.group("num")),
                     raw=text[s:e], start=s, end=e)
        )
        claimed.append((s, e))

    found.sort(key=lambda c: c.start)
    return _attach_pinpoint_prefixes(scan, text, found)


# "clause (9) of Article 22", "sub-clause (a) of clause (1) of Article 19",
# "sub-section (2) of section 41 of the CrPC". The pinpoint is written BEFORE
# the citation, so every pattern above matched only "Article 22" and the
# fabricated clause was silently dropped: the sentence then validated against
# a real article, a false VALIDATION of the D11 kind rather than a blind spot.
# Found in the 2026-10 fail-closed audit. The labels are moved onto the
# citation's key ("CONST§22(9)"), so MetadataDB.resolve checks them against
# the provision's own text exactly as it checks "Article 22(9)".
_PINPOINT_PREFIX = re.compile(
    r"(?:sub-?sections?|sub-?clauses?|clauses?|paragraphs?|provisos?\s+to)\s*"
    r"(?P<labels>(?:\(\s*[0-9a-zA-Z]{1,5}\s*\)\s*)+)of\s+(?:the\s+)?$",
    re.IGNORECASE)
_LABEL = re.compile(r"\(\s*([0-9a-zA-Z]{1,5})\s*\)")


def _attach_pinpoint_prefixes(scan: str, text: str, found: List[Citation]):
    starts = {}
    for c in found:
        starts[c.start] = starts.get(c.start, 0) + 1
    out = []
    for c in found:
        if starts[c.start] > 1:          # a list member: ambiguous, leave it
            out.append(c)
            continue
        start, outer_first = c.start, []
        while True:
            m = _PINPOINT_PREFIX.search(scan[max(0, start - 80):start])
            if not m:
                break
            # "sub-clause (a) of clause (1) of": the nearer prefix is the
            # outer label, so labels accumulate outermost-first.
            outer_first = _LABEL.findall(m.group("labels")) + outer_first
            start = max(0, start - 80) + m.start()
        if not outer_first:
            out.append(c)
            continue
        section = c.section + "".join(f"({l})" for l in _nest(outer_first))
        out.append(Citation(act=c.act, section=section, raw=text[start:c.end],
                            start=start, end=c.end))
    return out


def _nest(labels_read_order: List[str]) -> List[str]:
    """Labels as read left to right ("sub-clause (a) of clause (1) of") are
    innermost first; a key lists them outermost first ("19(1)(a)")."""
    return list(reversed(labels_read_order))


# ---------------------------------------------------------------------------
# BARE references: a section cited with NO act name -- "Section 41 permits
# arrest without a warrant [3]", "under section 7777 the court may...".
# Every pattern above requires an act name, so before the 2026-10 fail-closed
# audit a bare section was never extracted and therefore never checked: a
# fabricated "Section 7777 [1]" passed V2 untouched. Generators write bare
# sections constantly once the act has been named. The verifier resolves
# these against the act context (the acts of the chunks the sentence's
# markers point at, then the act most recently cited in the answer); a bare
# reference it cannot place in any act is stripped, not passed.
# ---------------------------------------------------------------------------
_BARE = re.compile(
    rf"(?:(?<!['\u2019])\b(?:sections?|secs?\.?|ss\.?|s\.|s(?=\s*\d)|u/s)|§§?)\s*"
    rf"(?P<list>{_NUM_ATOM}(?:{_SEP}{_NUM_ATOM})*)",
    re.IGNORECASE)
# Pinpoint with no section at all: "under sub-section (2)", "clause (b)".
_BARE_SUB = re.compile(
    r"\b(?:sub-?sections?|sub-?clauses?|clauses?)\s*"
    r"(?P<labels>(?:\(\s*[0-9a-zA-Z]{1,5}\s*\)\s*)+)(?!\s*of\s)",
    re.IGNORECASE)


_ACT_MENTION = re.compile(rf"\b(?P<act>{_ACT_PATTERN})(?!\w)", re.IGNORECASE)


def extract_act_mentions(text: str) -> List[tuple]:
    """(act code, offset) for every act the text names, with or without a
    section -- "Under the Criminal Procedure Code (CRPC) ... Section 41"
    places the bare "Section 41" in the CrPC."""
    out = []
    for m in _ACT_MENTION.finditer(_sanitise(text)):
        code = _resolve_act(m.group("act"))
        if code:
            out.append((code, m.start()))
    return out


@dataclass(frozen=True)
class BareReference:
    """A section (or sub-division) reference with no act named."""
    section: str        # "41", "41(1)(b)"; for a pinpoint-only ref, "(2)"
    raw: str
    start: int
    end: int
    pinpoint_only: bool = False


def extract_bare_references(text: str) -> List[BareReference]:
    scan = _sanitise(text)
    claimed = [(c.start, c.end) for c in extract_citations(text)]
    claimed += [(r.start, r.end) for r in extract_unresolved_act_refs(text)]
    out: List[BareReference] = []

    def free(s, e):
        return not any(not (e <= cs or s >= ce) for cs, ce in claimed)

    for m in _BARE.finditer(scan):
        s, e = m.span()
        if not free(s, e):
            continue
        nums = _expand_list(m.group("list"))
        # "sub-section (1) of section 102", "Clauses (3) of \u00a722": the same
        # written-before pinpoint as for a citation with its act named.
        labels, start = [], s
        if len(nums) == 1:
            while True:
                pm = _PINPOINT_PREFIX.search(scan[max(0, start - 80):start])
                if not pm:
                    break
                labels = _LABEL.findall(pm.group("labels")) + labels
                start = max(0, start - 80) + pm.start()
        for num in nums:
            sec = _clean_section(num) + "".join(f"({l})" for l in _nest(labels))
            out.append(BareReference(section=sec, raw=text[start:e],
                                     start=start, end=e))
        claimed.append((start, e))
    for m in _BARE_SUB.finditer(scan):
        s, e = m.span()
        if not free(s, e):
            continue
        labels = "".join(f"({l})" for l in _LABEL.findall(m.group("labels")))
        out.append(BareReference(section=labels, raw=text[s:e], start=s, end=e,
                                 pinpoint_only=True))
        claimed.append((s, e))
    out.sort(key=lambda r: r.start)
    return out


def normalise(surface: str) -> Optional[str]:
    """Normalise a single citation string to its canonical key, or None."""
    cits = extract_citations(surface)
    return cits[0].key if cits else None


# ---------------------------------------------------------------------------
# References to acts this corpus does not know.
#
# extract_citations() only recognises act names in ACT_ALIASES, so a section
# cited to ANY other act name produced zero citations and passed V2 unchecked:
# "Section 5 of the Indian Cyber Safety Act, 2019 makes phishing punishable"
# -- a statute that does not exist -- survived verification with only an
# advisory V5 note, while "Section 7777 IPC" (a fake section of a real act)
# was correctly stripped. A fabricated STATUTE NAME is at least as common a
# legal hallucination as a fabricated section number (a live Groq answer in
# this project's own experiment wrote "the Bharatiya Samvidhan Adhiniyam" for
# the Bharatiya Sakshya Adhiniyam), and it was the one form V2 could not see.
#
# This extracts section references whose act is NAMED (capitalised words
# ending in Act/Code/Sanhita/Adhiniyam/Rules/Regulations/Ordinance, or "Code
# of ...") but does not resolve to any act in the corpus. The verifier strips
# them for the same reason it strips any other citation it cannot resolve:
# the guarantee is that every emitted citation resolves to the corpus, and an
# act outside it -- real (the Code of Civil Procedure) or invented -- cannot
# be checked. Generic back-references with no name ("section 3 of the Act",
# "of the said Act") are not matched.
# ---------------------------------------------------------------------------

_ACT_KEYWORD = (r"(?:Act|Code|Sanhita|Adhiniyam|Rules|Regulations|Ordinance"
                r"|Law|Bill)")
_NAME_WORD = r"[A-Z][A-Za-z'\u2019\-]*\.?"
_CONNECTOR = r"(?:of|and|for|the|on|to|in|from|&)"
_ACT_NAME = (rf"(?:{_NAME_WORD}\s+(?:{_CONNECTOR}\s+)*){{1,8}}{_ACT_KEYWORD}\b"
             rf"|Code\s+of\s+(?:{_NAME_WORD}\s*){{1,4}}")
# Case-insensitive variant, used ONLY after "section N of (the)": there, "act"
# can only be a statute ("section 5 of the indian cyber safety act"), whereas
# in free prose "act" is also a verb ("to act under section 5"), so the
# act-first form keeps requiring capitals.
_ACT_NAME_CI = (rf"(?:[A-Za-z][A-Za-z'\u2019\-]*\.?\s+(?:{_CONNECTOR}\s+)*){{1,8}}"
                rf"(?i:{_ACT_KEYWORD})\b"
                rf"|(?i:code\s+of)\s+(?:[A-Za-z][A-Za-z'\u2019\-]*\s*){{1,4}}")
# Words that make "of the <word> Act" a back-reference to some act already in
# context rather than a NAME: "the said Act", "this Sanhita", "the new Act".
_GENERIC_ACT_WORDS = {
    "the", "a", "an", "this", "that", "said", "same", "such", "principal",
    "parent", "above", "aforesaid", "new", "old", "present", "relevant",
    "amending", "repealed", "original", "any", "every", "each", "other",
    "respective", "concerned", "applicable", "corresponding", "central",
    "state", "which", "whose", "its", "their",
}
_SEC_WORD_CS = r"(?i:\b(?:sections?|sec\.?|ss\.?|s\.|u/s|under\s+section))"

_UNRES_SEC_FIRST = re.compile(
    rf"{_SEC_WORD_CS}\s*\(?\s*(?P<num>{_NUM})\s*\)?{_FOOTNOTE}\s*(?:,\s*)?"
    rf"(?i:of)\s+(?:(?i:the)\s+)?(?P<name>{_ACT_NAME_CI})")
# Indian statutes are also cited by number: "Manipur Act 3 of 1985, Section
# 4(2)". The optional "N of YYYY" group keeps that form from escaping.
_ACT_NUMBER = r"(?:\s+\d+\s+of\s+\d{4})?"
_UNRES_ACT_FIRST = re.compile(
    rf"(?P<name>{_ACT_NAME}){_ACT_NUMBER}\s*(?:,\s*(?:\d{{4}}\s*,?\s*)?)?\(?\s*"
    rf"{_SEC_WORD_CS}\s*(?P<num>{_NUM})")


@dataclass(frozen=True)
class UnresolvedActRef:
    """A section reference whose named act is not in the corpus."""
    act_name: str
    section: str
    raw: str
    start: int
    end: int


def _is_generic(name: str) -> bool:
    """A back-reference ("the said Act", "this Sanhita") rather than a name."""
    words = [w.lower().strip(".,") for w in name.split()]
    while words and words[0] == "the":
        words = words[1:]
    return len(words) < 2 or words[0] in _GENERIC_ACT_WORDS


def _name_resolves(name: str) -> bool:
    words = name.strip().split()
    if words and words[0].lower() == "the":
        words = words[1:]
    # Any word-prefix resolving means the name is (or begins with) a known
    # act: "Code of Criminal Procedure Section" must not be flagged just
    # because the greedy name pattern ran one word too far.
    for i in range(len(words), 0, -1):
        if _resolve_act(" ".join(words[:i]).rstrip(",")):
            return True
    return False


def act_names_before(text: str, pos: int) -> Optional[tuple]:
    """The act named nearest before `pos`: (code, end) for a corpus act,
    (None, end) for a NAMED act outside the corpus ("the Indian Telegraph
    Act"), or None. A bare "\u00a7\u00a7 5-7" right after "the Indian Telegraph Act,
    1885" belongs to that act -- not to the Information Technology Act named
    after it, which is where it was placed before this check (found by the
    independent audit of the second live round)."""
    scan = _sanitise(text[:pos])
    best = None
    for m in _ACT_MENTION.finditer(scan):
        code = _resolve_act(m.group("act"))
        if code and (best is None or m.end() > best[1]):
            best = (code, m.end())
    for m in re.finditer(_ACT_NAME, scan):
        name = m.group(0).strip()
        if _is_generic(name) or _name_resolves(name):
            continue
        if best is None or m.end() > best[1]:
            best = (None, m.end())
    return best


def extract_unresolved_act_refs(text: str) -> List[UnresolvedActRef]:
    return list(_unresolved_cached(text))


@lru_cache(maxsize=8192)
def _unresolved_cached(text: str):
    return tuple(_extract_unresolved_act_refs(text))


def _extract_unresolved_act_refs(text: str) -> List[UnresolvedActRef]:
    scan = _sanitise(text)
    claimed = [(c.start, c.end) for c in extract_citations(text)]
    out: List[UnresolvedActRef] = []
    for pat in (_UNRES_SEC_FIRST, _UNRES_ACT_FIRST):
        for m in pat.finditer(scan):
            s, e = m.span()
            if any(not (e <= cs or s >= ce) for cs, ce in claimed):
                continue
            name = m.group("name").strip()
            if _is_generic(name) or _name_resolves(name):
                continue
            out.append(UnresolvedActRef(act_name=name,
                                        section=_clean_section(m.group("num")),
                                        raw=text[s:e], start=s, end=e))
            claimed.append((s, e))
    out.sort(key=lambda r: r.start)
    return out
