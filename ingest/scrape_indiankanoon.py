"""
Parser for Indian Kanoon's bare-act HTML into structured provisions.

SOURCE AND WHY. India Code (indiacode.nic.in) is the official government
repository and the paper's stated preferred source, but every content path on
it (handle pages, PDF bitstreams, an alternate render endpoint) returned
403/404 to automated requests from this environment while the bare domain
returned 200 -- a bot-protection block, confirmed across four independently
sourced URLs across two different acts, not a one-off fluke or a wrong URL.
CLAUDE.md's own acquisition note says "academic pre-packaged distributions are
preferred to direct scraping wherever available" -- consistent with that,
Indian Kanoon (indiankanoon.org) was used instead: a long-established, publicly
reachable Indian legal database whose robots.txt permits general crawling
(it disallows a specific list of individually-requested document takedowns,
not statutory text), and which is the underlying source for several corpora
this paper already cites (ILDC, IL-TUR). Its markup follows the Akoma Ntoso
legislative-XML convention, which makes structured parsing reliable rather
than fragile regex over flowing prose.

A separate, commercially-published commentary PDF was found and REJECTED as a
source during this work: it interleaves case-law annotations and footnotes
into the statutory text and is almost certainly under separate copyright from
the bare Act. Using it would risk exactly the kind of misattributed content
this project exists to prevent. See the fetch log / project history for the
rejection reasoncontent -- it is not referenced anywhere in this pipeline.

WHAT THIS PRODUCES. Real, verbatim, government-enacted statutory text (public
domain under S.52(1)(q) of the Copyright Act, 1957, per this project's existing
ethics note), structured as (act, section, marginal_note, text). Nothing here
is paraphrased -- every provision produced by this module is marked
verbatim=True, in contrast to data/seed_corpus.py's hand-typed paraphrases.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


@dataclass
class ScrapedSection:
    number: str          # "420", "103(1)", "19(1)(a)"
    marginal_note: str
    text: str
    source_url: str


# Each entry: (internal name, cached filename, canonical act code, source URL,
# the numbering word used for citations of this act -- "Section" for statutes,
# "Article" for the Constitution).
ACT_SOURCES = [
    ("ipc",   "ipc.html",   "IPC",   "https://indiankanoon.org/doc/1569253/",  "Section"),
    ("bns",   "bns.html",   "BNS",   "https://indiankanoon.org/doc/149679501/", "Section"),
    ("crpc",  "crpc.html",  "CRPC",  "https://indiankanoon.org/doc/445276/",   "Section"),
    ("bnss",  "bnss.html",  "BNSS",  "https://indiankanoon.org/doc/91117739/", "Section"),
    ("iea",   "iea.html",   "IEA",   "https://indiankanoon.org/doc/1953529/",  "Section"),
    ("bsa",   "bsa.html",   "BSA",   "https://indiankanoon.org/doc/70224818/", "Section"),
    ("nia",   "nia.html",   "NIA",   "https://indiankanoon.org/doc/1132672/",  "Section"),
    ("ita",   "ita.html",   "ITA",   "https://indiankanoon.org/doc/1965344/",  "Section"),
    ("ica",   "ica.html",   "ICA",   "https://indiankanoon.org/doc/171398/",   "Section"),
    ("const", "const.html", "CONST", "https://indiankanoon.org/doc/237570/",   "Article"),
]

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    # U+00AD SOFT HYPHEN: an invisible print-only hyphenation hint the
    # source PDF-to-HTML pipeline left embedded mid-word (IPC \S31:
    # "docu\xadment" for "document") -- invisible when rendered, but a
    # real, distinct character in the string. Found via corpus-wide scan:
    # 137 occurrences across the ten Acts, every one splitting a real word
    # this way. Left in, it silently breaks exact-token quotation
    # matching (V4) for any claim that quotes the affected word normally
    # (without the hyphen) -- the two strings look identical to a human
    # reader and differ only in an invisible character, exactly the kind
    # of gap this project's own quotation checker exists to not have.
    # Deleted outright rather than replaced with anything, since -- unlike
    # \xa0, which stands in for a real space -- a soft hyphen has no
    # visible-character equivalent; it marks a hyphenation point, not a
    # character in the word itself.
    text = text.replace("\xad", "")
    return _WS.sub(" ", text).strip()


# STATE / UNION-TERRITORY AMENDMENT TABLES. Indian Kanoon renders each state's
# local amendments to a central Act ("UTTAR PRADESH.- In Section 436 ...",
# "ANDAMAN AND NICOBAR ISLANDS (UT).- After Section 3 ...") as a table inside
# the central section, and those tables were left in the section's body -- so
# law that applies in ONE state was presented as the nationally operative
# text: a fabricated quotation drawn from Maharashtra's s.438 would validate
# against a citation to the central CrPC s.438 (V4), and IEA s.90A in the
# corpus was actually Uttar Pradesh's s.90-A, a different presumption
# entirely, because a heading inside the UP table was split out as if it were
# the central section.
#
# The rule: a table whose text BEGINS with a state or union-territory name
# (or "In its application to the State of ...") is a state amendment.
# Genuine schedule tables begin with column headers ("Offence. Section of the
# Indian Penal Code applicable...", "Name Territories", "Sl. No.") or row
# numbers. Checked against all 189 tables in the ten Acts: 60 classified as
# state amendments (58 CrPC, 2 IEA), every one of them read and confirmed to
# be one; none of the 129 kept tables is. Known misses, left in as before: a
# few state rows appended to the CrPC s.320 compounding table open with a
# quoted row rather than a state name.
_STATE_NAMES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "uttaranchal", "west bengal", "andaman and nicobar", "andman and nicobar",
    "chandigarh", "dadra and nagar haveli", "daman and diu", "delhi",
    "jammu and kashmir", "ladakh", "lakshadweep", "puducherry", "pondicherry",
    "mysore", "madras", "bombay", "east punjab", "u.p.", "m.p.", "w.b.",
    "a.p.", "h.p.",
]
_STATE_START = re.compile(
    r"^\s*(?:the\s+)?(?:"
    + "|".join(re.escape(n) for n in sorted(_STATE_NAMES, key=len, reverse=True))
    + r")\b", re.IGNORECASE)
_APPLICATION_TO_STATE = re.compile(
    r"^\s*in its application to the (?:state|union territory)", re.IGNORECASE)


def is_state_amendment_table(text: str) -> bool:
    return bool(_STATE_START.match(text) or _APPLICATION_TO_STATE.match(text))


def _body_text(tag) -> str:
    """
    Full body text of a section tag, excluding its own heading and any
    editorial commentary.

    A THIRD markup variant, beyond the two documented at module level, was
    found in the IT Act: a lead-in paragraph in `akn-intro` followed by
    enumerated sub-clauses nested in child `akn-paragraph` sections, each with
    its own `akn-content`. Taking only the first `akn-content` found by a
    subtree search (the previous approach) silently returned just the FIRST
    such clause -- Section 66 ("Computer related offences") resolved to only
    its explanation clause (a), discarding the substantive offence-defining
    sentence and clause (b) entirely. This produces a coherent-looking but
    WRONG provision: well-formed prose, plausible marginal note, incorrect
    body -- caught only by cross-checking scraped text against a section whose
    real content was already known, which is why that check is retained as an
    automated test rather than a one-off spot check.

    The fix takes the full text of the section's subtree indiscriminately --
    akn-content, akn-intro, and nested akn-paragraph clauses are all just text
    at that point -- MINUS anything Indian Kanoon marks as its own added
    commentary (`data-status="editorial"`), which is not part of the enacted
    text and would otherwise corrupt exact-quotation matching. We reparse a
    string copy of the tag rather than mutate the live tree, since mutation
    would remove nodes the sibling-walk in parse_akoma_html still needs.
    """
    from bs4 import BeautifulSoup as _BS
    work = _BS(str(tag), "html.parser")
    for h in work.find_all(["h2", "h3", "h4"]):
        h.decompose()
    for rem in work.find_all(attrs={"data-status": "editorial"}):
        rem.decompose()
    # A THIRD commentary artefact, found by the corpus-wide structural
    # audit: Indian Kanoon renders a repealed/superseded sub-section's OLD
    # text -- not wrapped in a data-status="editorial" span, but in a
    # plain `akn--table-container` table sitting right inside the current
    # section -- e.g. IEA \S119's table opens "Prior to substitution by
    # Criminal Law (Amendment) Act, 2013 section 119 read as;119. Dumb
    # witnesses.- A witness who is unable to speak..." followed by the
    # ENTIRE pre-2013 text of the sub-section. Left in, this silently
    # pastes non-current statutory text into what's presented as \S119's
    # operative body -- a citation quoting the current, in-force text is
    # unaffected, but a fabricated quote drawn from the superseded wording
    # would incorrectly validate against it, exactly the failure mode V4
    # exists to prevent. Confirmed at CRPC \S\S54, 127, 309 and IEA \S\S119,
    # 114A this way; only tables matching this specific, unambiguous
    # marker phrase are removed -- NOT decided here is the broader,
    # fuzzier question of state/UT-specific amendment tables (e.g. "In its
    # application to the State of Andhra Pradesh...") sharing the same
    # `akn--table-container` markup, which a corpus-wide check found
    # mixed in among genuinely-current, substantive schedule tables (the
    # CRPC \S320 compounding-of-offences table, the classification-of-
    # offences First Schedule) with no single reliable phrase distinguishing
    # them from those. That gap is now closed by is_state_amendment_table
    # (below the function): see its comment for the rule and how it was
    # checked against every table in the ten Acts.
    _PRIOR_VERSION = re.compile(
        r"prior to (its )?substitution|read as under\b|"
        r"section \d+[A-Za-z]* read as\b", re.IGNORECASE)
    for tbl in work.find_all("div", class_="akn--table-container"):
        head = tbl.get_text(separator=" ", strip=True)[:150]
        if _PRIOR_VERSION.search(head) or is_state_amendment_table(head):
            tbl.decompose()
    # A second, separate commentary artefact: a "References" mini-list of
    # links to external sites (IndianKanoon's own further-reading pointers,
    # not the enacted text), living OUTSIDE the editorial-remark span the step
    # above removes. Its links carry class `akn-ref`; removing them and any
    # bare "References" label left behind covers both an editorial comment
    # and a references list appearing independently of one another.
    for ref in work.find_all("a", class_="akn-ref"):
        ref.decompose()
    # separator=" ": adjacent list-clause spans in the source markup
    # routinely carry NO whitespace between them at all --
    # "<span>...or</span><span>is of such a nature...</span>" -- and
    # get_text()'s default (no separator) glues them into "orIS OF such a
    # nature" verbatim, corrupting text this project's own quotation check
    # (V4) exists to verify exactly. _clean() immediately below collapses
    # any resulting double-space back to one, so this is safe wherever the
    # source already had whitespace between elements; it only adds
    # whitespace that was otherwise missing.
    text = _clean(work.get_text(separator=" "))
    text = re.sub(r"\bReferences\s*$", "", text).strip()
    # A FOURTH shape of the same prior-version pollution _PRIOR_VERSION
    # (above) removes from table markup: CRPC \S127's sub-section (1) has
    # the identical "[[Substituted by ... Prior to its substitution ...
    # read as under :- (old text) ]]" notice sitting inline in plain
    # paragraph prose, no table involved at all. Removed the same way
    # `_heading_text` truncates a heading at this marker -- except here
    # real current text can legitimately follow the noise within the same
    # subsection, so this only strips the bracketed span itself (from the
    # opening amendment marker through its own closing bracket run)
    # rather than truncating everything after it.
    text = re.sub(
        r"\[+\s*(?:Substituted|Inserted|Omitted|Renumbered|Repealed)\s+by\b"
        r".*?read as under\s*:?-?.*?\]+",
        "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = _WS.sub(" ", text)
    return text


_AMENDMENT_MARKER = re.compile(
    r"\[+\s*(?:Substituted|Inserted|Omitted|Renumbered|Repealed)\s+by\b",
    re.IGNORECASE)


def _heading_text(h) -> str:
    r"""
    Heading text with the same editorial-remark stripping `_body_text`
    already applies to body text, applied to the `<h3>`/`<h4>`/`<h2>` tag
    instead. Found by the corpus-wide structural audit: a heading whose
    marginal note reads far longer than any real Act heading is a section
    that was amended or inserted after enactment, e.g. CRPC \S166A --
    ``<h3>166A. [ Letter of request... <span data-status="editorial">
    [Inserted by Act 10 of 1990, ...]</span></h3>`` -- whose marginal note
    then reads ``[ Letter of request ... [Inserted by Act 10 of 1990,
    Section 2 (w.e.f. 19-2-1990).]``, pollution `_body_text` was already
    built to keep OUT of the enacted text but that this parallel path never
    filtered, because it calls `h.get_text()` directly rather than going
    through `_body_text`'s cleanup. Reparses a string copy for the same
    reason `_body_text` does: mutating the live tag would remove nodes nothing
    else needs here, but doing it via a copy costs nothing and matches the
    existing pattern rather than adding a second one.

    A second, more severe case of the SAME underlying problem is not
    wrapped in a `data-status="editorial"` span at all: CRPC \S330's raw
    heading text is ``330. [ Release of person of unsound mind pending
    investigation or trial. [[Substituted by the Code of Criminal Procedure
    (Amendment) Act, 2008 ... Prior to its substitution, Section 330 read as
    under :- 330. Release of lunatic pending investigation or trial. -
    (1) Whenever a person is found ...'' -- the ENTIRE pre-amendment text of
    the section, unmarked, dumped directly into the heading with nothing but
    a bracket for a human reader to notice. Confirmed to affect exactly one
    heading across all ten scraped Acts (a corpus-wide search for the same
    "read as under" phrasing found fourteen other occurrences, all in body
    prose elsewhere, not headings). Truncated at the first occurrence of a
    bracketed amendment marker ("[Substituted by", "[[Inserted by", ...).
    Both this case and the editorial-span case leave a dangling leading
    "[" in the section number's marginal note afterward -- stripped by
    `_match_leading_number`, which is where the note is actually carved out
    from the number, not here (this function's own return value still
    starts with the number itself).
    """
    from bs4 import BeautifulSoup as _BS
    work = _BS(str(h), "html.parser")
    for rem in work.find_all(attrs={"data-status": "editorial"}):
        rem.decompose()
    text = _clean(work.get_text(separator=" "))
    m = _AMENDMENT_MARKER.search(text)
    if m:
        text = text[: m.start()].strip()
    # NOT lstrip("[") here: the returned text still starts with the section
    # number ("166A. [ Letter..."), so a leading "[" belongs to the NOTE
    # that comes after the number, not to this string's own first
    # character -- stripped instead by _match_leading_number, which is
    # where the note is actually carved out from the number.
    return text


_TITLE_BODY_SPLIT = re.compile(r"^(.{3,120}?)—\s*(\S.*)$", re.DOTALL)

# A note this long is not a title any more -- every genuine marginal note
# sampled across the ten scraped Acts (including six that legitimately
# contain an em dash as part of a short title, e.g. IPC \S70 "Fine leviable
# within six years, or during imprisonment -- Death not to discharge
# property from liability", 103 chars) stays under 120 chars. ICA \S68's
# heading-embedded operative rule runs 381. 150 gives headroom above every
# observed legitimate note while still catching the bug cases.
_LONG_NOTE_THRESHOLD = 150


def _rescue_heading_only_section(current: dict, body: str) -> str:
    """
    Some sections carry their ENTIRE content -- not just a long marginal
    note, the actual operative rule -- directly inside the heading tag,
    rather than in a separate content container the way every ordinary
    section does: `<section class="akn-section" id="section_184"><h3>184.
    Who may be an agent—As between the principal and third person any
    person may become an agent...</h3></section>`, no `akn-content`, no
    `akn-subsection`, nothing else at all. `_body_text` decomposes
    h2/h3/h4 before extracting text specifically so a section's own
    heading is never counted as its body -- correct for every ordinary
    section, where the real content lives elsewhere in the tag -- but
    here that leaves `_body_text` returning an empty string, and
    `flush()`'s `if body and ...` check then drops the section ENTIRELY,
    silently: not a wrong section, an invisible one, worse than any
    defect this project had found before it (a fabricated citation is at
    least checkable against something; a citation to a real section this
    parser dropped resolves to "does not exist," indistinguishable from a
    genuine fabrication). Confirmed at ICA \\S184 this way.

    ICA \\S68 is the same bug in a milder disguise: the operative rule is
    still stuck in the heading, but the section ALSO has a separate
    "Illustrations" block, so `_body_text` returns non-empty (the
    illustrations alone) and the empty-body check above never fires --
    the section survives, silently missing its actual rule. Caught
    instead by note length: legitimate marginal notes, even the six in
    this corpus that use an em dash themselves, all stay under 120 chars
    (see `_LONG_NOTE_THRESHOLD`); ICA \\S68's runs 381.

    Indian statutory drafting conventionally separates a short title from
    the rule that follows it with an em dash ("Title.—Operative text
    ..."), which `_match_leading_number` already captures whole into
    `current["note"]` since it has no way to know part of that note is
    actually the body. Split there when the pattern is clean, narrowing
    the note back down to a real title and returning the rest to be
    prepended ahead of whatever real body content (e.g. ICA \\S68's
    illustrations) already exists. A long note with no clean split (IPC
    \\S56, a 1949 repeal notice with no operative rule to separate out) has
    no title to narrow to; used as the body too, but ONLY when body is
    otherwise empty, since with real body content already present
    elsewhere a messy unsplit note is more likely noise than a rescue.
    """
    note = current["note"]
    if len(note) <= _LONG_NOTE_THRESHOLD:
        return ""
    m = _TITLE_BODY_SPLIT.match(note)
    if m:
        title, rescued = m.group(1).strip(), m.group(2).strip()
        current["note"] = title
        return rescued
    if not body:
        return note
    return ""


_BARE_SUBSECTION_NUM = re.compile(r"^\(?[0-9]{1,3}[A-Za-z]?\)?\.?$")
# A second, distinct signature of the same orphaned-sibling bug: a clause
# introducing an enumerated list ("Whoever commits mischief by,-", "shall,-")
# whose own body ends at the comma/dash rather than continuing into the
# list, because the list itself lives in the same kind of sibling span the
# bare-numeral case does. Found the same way -- BNS§326's body was
# "Whoever commits mischief by," with the (a)/(b)/(c) list one sibling over.
# Capped at 150 chars so a legitimately short, complete sentence that
# happens to contain an internal comma near the end is not misread as
# truncated; every confirmed case so far is well under that.
_TRUNCATED_LIST_INTRO = re.compile(r",\s*-?\s*$")


def _resolve_orphaned_subsection(tag, body: str) -> str:
    """
    A SIXTH markup variant, found via the corpus-wide structural audit in
    Section~VIII rather than a spot check or a generation experiment: some
    `akn-subsection` tags contain ONLY their numeral
    (`<span class="akn-num">1</span>`) with no nested content at all -- the
    actual prose lives in a following SIBLING span (`akn-hcontainer` for
    most subsections, `akn-wrapUp` for a section's last one; found to be two
    different class names for the same structural pattern, so this checks
    for the presence of an `akn-p` inside the sibling rather than hardcoding
    either name, in case a third variant of the same wrapper exists
    elsewhere). Constitution Article 334A (reservation of seats for women)
    is the confirmed case: all four of its subsections used this pattern,
    and `_body_text` alone -- which only walks TAG's own subtree -- returned
    just "1", "2", "3", "4", losing the entire ~1,600 characters of
    substantive text.

    A second signature of the identical structural bug, found on the same
    audit pass: an enumerated-list-introducing clause whose own body ends at
    the comma/dash rather than continuing into the list -- BNS§326
    ("Mischief by injury..."), whose body resolved to "Whoever commits
    mischief by," with clauses (a)/(b)/(c) one sibling over, unreachable the
    same way the bare numeral was.

    Not every bare-numeral or comma-ending subsection is this bug (most
    Constitution articles legitimately have a short numeral immediately
    followed by nested content in the ordinary way, and most sentences that
    happen to contain a comma are simply complete); this only fires when the
    tag's own body resolved to nothing but the numeral or a truncated list
    intro AND a content-bearing sibling actually exists to rescue it from --
    a body with no such sibling is returned unchanged, so this can only add
    text that was genuinely missing, never alter a body that was already
    complete. Pulls only from an immediate span sibling, stopping at the
    next `<section>` boundary so it can never cross into a different
    subsection or section's content.
    """
    if not body:
        return body
    stripped = body.strip()
    if not (_BARE_SUBSECTION_NUM.match(stripped) or
           (len(stripped) < 150 and _TRUNCATED_LIST_INTRO.search(stripped))):
        return body
    sib = tag.find_next_sibling()
    hops = 0
    while sib is not None and hops < 3:
        if getattr(sib, "name", None) == "section":
            break
        if getattr(sib, "name", None) == "span" and sib.find("span", class_="akn-p") is not None:
            extra = _body_text(sib)
            if extra:
                return f"{body} {extra}".strip()
            break
        sib = sib.find_next_sibling()
        hops += 1
    return body


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _dedup_parts(parts: List[str]) -> List[str]:
    """
    Drop individually-appended subsection parts that the SECTION-level grab
    (`parts[0]`) already contains in full, keeping only whatever tail is
    genuinely new.

    A SEVENTH markup variant, the third found by the same corpus-wide audit
    that found the sixth: for the "nested" markup variant (IPC, CRPC, IEA,
    BSA, NIA, ICA, the Constitution -- Section~VIII), subsections are
    children of their parent `akn-section` tag, so `_body_text` on the
    SECTION tag already walks into them and returns the complete text --
    and the main loop then ALSO processes each child `akn-subsection` tag
    on its own and appends ITS body again. Most of the time this produces
    an exact whole-body duplicate that `_dedup_halves`'s tail search
    already collapses (that function's original purpose). BNSS§266 exposed
    the case that breaks the tail-only assumption: subsection (3)'s own
    nested body was orphaned (three bare characters, rescued by
    `_resolve_orphaned_subsection` to its full text) -- content the
    section-level grab never had at all, sitting AFTER two subsections'
    worth of pure duplication. The true duplicate is therefore not at the
    tail of the joined string; it is parts[0] against parts[1] and
    parts[2], with part 3 as a genuinely unique tail `_dedup_halves` cannot
    tell apart from more duplication using string position alone.

    Operating on the parts BEFORE joining sidesteps that ambiguity
    entirely: walk parts[1:], accumulating their tokens, for as long as
    the accumulation is a token-for-token PREFIX of parts[0]'s own tokens
    -- that span is definitionally redundant, since the parser already
    captured it once. The moment a part's tokens stop matching (or the
    accumulation exceeds parts[0]'s length), stop consuming: everything
    from there on, redundant or not, is left for `_dedup_halves` to
    resolve on the joined string exactly as it already does today, so
    provisions that never hit this variant see no change in behaviour.
    """
    if len(parts) < 2:
        return parts
    head = _TOKEN.findall(parts[0].lower())
    acc: List[str] = []
    cut = 0
    for i, part in enumerate(parts[1:], start=1):
        acc.extend(_TOKEN.findall(part.lower()))
        if len(acc) > len(head):
            break
        if head[: len(acc)] == acc:
            cut = i
            if len(acc) == len(head):
                break
        else:
            break
    if cut == 0:
        return parts
    return [parts[0]] + parts[cut + 1:]


def _dedup_halves(body: str) -> str:
    """
    Collapse a body that ends in a duplicate of its own preceding content.

    A fourth artefact, found in heavily-amended Constitution articles (Article
    19, whose sub-clauses were individually amended across multiple
    constitutional amendments): the page renders the same clause text twice
    under different internal groupings -- once compactly nested with no
    inter-clause spacing, once again via sibling akn-subsection tags with
    normal spacing. Both get collected by the walk above, so the resulting
    body is the true text immediately followed by a whitespace-normalised
    copy of itself.

    Originally this checked only an exact 50/50 split -- correct for Article
    19, where the WHOLE body is two copies back to back. IEA§146 (found by
    the same corpus-wide audit that found the orphaned-subsection defect)
    exposed the general case: the duplicate is a TAIL of the body, not the
    whole thing, because the intro clause ("...which tend -") is captured
    only once while the enumerated list after it is captured twice -- an
    unequal split the exact-half check cannot see, silently leaving the
    duplication in place. This tries every candidate split from the largest
    (the original exact-half case, tried first so behaviour there is
    unchanged) down to a floor of 15 tokens, so a coincidentally-repeated
    short phrase is never mistaken for this artefact. Detected by
    token-level comparison (spacing differs between the copies, so a
    literal string bisection would miss it either way) and collapsed to a
    single copy; left alone if no such duplication is present at any split.
    """
    matches = list(re.finditer(_TOKEN, body.lower()))
    toks = [m.group(0) for m in matches]
    n = len(toks)
    for k in range(n // 2, 15, -1):
        if 2 * k > n:
            continue
        if toks[n - k:] == toks[n - 2 * k: n - k]:
            cut = matches[n - k].start()
            # The token at the cut boundary doesn't include whatever
            # punctuation immediately precedes it (an opening "(" or "["
            # belonging to the removed copy, e.g. "(1)" is not itself a
            # token) -- strip that too, or it dangles off the kept prefix.
            trimmed = body[:cut].rstrip()
            trimmed = re.sub(r"[\s(\[-]+$", "", trimmed).rstrip()
            return trimmed
    return body


def _match_leading_number(heading: str) -> Optional[dict]:
    """
    Extract {"number", "note"} from a heading, or None if it doesn't start
    with a section number. The number is taken by regex rather than by
    assuming an anchor tag wraps it -- not every heading has one (Article
    334A's does not), and taking the anchor-less fallback (the WHOLE heading
    text) as the "number" swallows the marginal note into it too.
    """
    m = re.match(r"^(\d+[A-Za-z]*)\.?\s*", heading)
    if not m:
        return None
    # A leading "[" here (not stripped by _heading_text, which only sees
    # the number-inclusive heading and so never has "[" as its own first
    # character) marks a section that was substituted or inserted after
    # enactment -- Indian Kanoon's own convention for flagging that, not
    # part of the marginal note itself. _heading_text already removes the
    # amendment-history text that used to follow it; this removes the
    # bracket its removal leaves dangling.
    note = heading[m.end():].strip().rstrip("—-. ").strip()
    note = note.lstrip("[").strip()
    # The leading "[" just stripped is often one half of a pair wrapping
    # only the note's own title clause, not a marker followed by more
    # note text -- e.g. IPC \S56 "[Sentence of Europeans...for life.] Rep.
    # by the Criminal Law...": the trailing "]" is that same bracket's
    # closer, now orphaned. Corpus-wide check (all ten Acts) found this
    # exact "text.] Rep. by ..." shape at IPC \S\S56, 58, 161, 165, 492 --
    # five repealed sections, all with the identical dangling artifact.
    # Only fire when the note has a "]" with NO surviving "[" at all: a
    # note that still has both (IEA \S34's "...electronic form] , when
    # relevant", CONST's "... [Repealed]") is a BALANCED, legitimate
    # bracket the drafting itself uses and must not be touched.
    if "]" in note and "[" not in note:
        note = note.replace("]", "", 1).strip()
    return {"number": m.group(1), "note": note}


def parse_akoma_html(path: str, source_url: str) -> List[ScrapedSection]:
    """
    Parse one Indian Kanoon bare-act page.

    Three markup variants were found across the nine acts scraped:

    1. Body nested directly in the section (`akn-content`) -- IPC, CRPC, IEA,
       BSA, NIA, ICA, the Constitution.
    2. Body as SIBLING `akn-subsection` elements following an otherwise-empty
       section tag -- BNS. Handling only variant 1 silently produced zero
       sections for BNS, caught by checking the count against a known figure.
    3. A lead-in `akn-intro` paragraph plus nested `akn-paragraph` sub-clauses,
       each with their own `akn-content` -- the IT Act. Taking the first
       `akn-content` match anywhere in the subtree (an earlier version of this
       parser) silently returned only the first sub-clause: syntactically
       fine, semantically wrong. See `_body_text` for the fix.

    We walk every akn-section/akn-subsection tag in DOCUMENT ORDER
    (`soup.find_all` returns them depth-first, i.e. document order regardless
    of nesting) and attribute each trailing subsection's text to whichever
    section most recently opened, which is correct under all three variants.
    """
    with open(path, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    tags = soup.find_all("section",
                         class_=("akn-section", "akn-subsection", "akn-chapter"))

    out: List[ScrapedSection] = []
    seen_numbers = set()
    current = None          # in-progress ScrapedSection fields
    current_body_parts: List[str] = []

    def flush():
        if current is None:
            return
        parts = _dedup_parts(current_body_parts)
        body = _clean(" ".join(parts))
        body = _dedup_halves(body)
        # A per-element strip of a trailing "References" label (in
        # _body_text) does not catch every case: when the References mini-list
        # arrives as its OWN separate sibling segment, joining it onto the
        # preceding segment first is what produces the trailing label -- the
        # per-element check on that segment alone can still miss it depending
        # on what else shares the segment. Stripped again here, after joining,
        # closes that gap.
        body = re.sub(r"\bReferences\s*$", "", body).strip()
        num = current["number"]
        if body and num not in seen_numbers:
            seen_numbers.add(num)
            out.append(ScrapedSection(number=num, marginal_note=current["note"],
                                      text=body, source_url=source_url))

    for tag in tags:
        classes = tag.get("class") or []
        if "akn-section" in classes:
            h = tag.find(["h3", "h4"])
            if h is None:
                continue
            heading = _heading_text(h)
            new_section = _match_leading_number(heading)
        elif "akn-chapter" in classes:
            # A FOURTH markup variant, found in BNS specifically: section 318
            # ("Cheating") is rendered as TWO <section class="akn-chapter">
            # tags -- "Chapter Of cheating" then "Chapter 318. Cheating." --
            # instead of the usual akn-section/h3 pattern. It was invisible to
            # every version of this parser until now, because akn-chapter was
            # never in the collected tag set at all: not a total miss like the
            # original BNS defect (that produced zero sections; this produced
            # 351 CORRECT sections and one section whose content was silently
            # merged onto its numeric predecessor -- section 317's scraped
            # body ran on into what is actually section 318's text). Found
            # only by reading real generated output against the actual source
            # HTML during the generation-experiment build, not by the
            # structural sweep, which had no way to know 317's body was too
            # long. Most akn-chapter tags are genuine chapter dividers
            # ("CHAPTER I", "OF OFFENCES AGAINST PROPERTY") and must NOT be
            # treated as sections; only ones whose heading itself matches the
            # numbered-section pattern after a "Chapter " prefix are this
            # mislabelled case, confirmed to occur exactly once across all ten
            # scraped Acts.
            h = tag.find("h2")
            if h is None:
                continue
            heading = _heading_text(h)
            m = re.match(r"^Chapter\s+(\d+[A-Za-z]*\.\s*.+)$", heading)
            new_section = _match_leading_number(m.group(1)) if m else None
        else:
            new_section = None

        if new_section:
            flush()
            current, current_body_parts = new_section, []
            body = _body_text(tag)
            # Some sections dump their entire operative rule into the
            # heading itself rather than a separate content container
            # (ICA \S184: nothing else in the tag at all; ICA \S68: a
            # separate "Illustrations" block exists, so `body` is non-empty
            # but still missing the actual rule; IPC \S56: a repeal notice,
            # same shape). Rescue it before it's lost -- see
            # _rescue_heading_only_section for what each case needs.
            rescued = _rescue_heading_only_section(current, body)
            if rescued:
                current_body_parts.append(rescued)
            if body:
                current_body_parts.append(body)
        elif "akn-subsection" in classes and current is not None:
            body = _body_text(tag)
            body = _resolve_orphaned_subsection(tag, body)
            if body:
                current_body_parts.append(body)

    flush()
    return _split_embedded_subsections(out, path, source_url)


# Run-in heading of an inserted section: "65B. Admissibility of electronic
# records." Seen in the source in several forms, each of which the earlier
# pattern (digits + letter, no separator, title ending in a full stop)
# missed, leaving the inserted section merged into its parent: "113-B."
# (IEA 113-A, 113-B -- the dowry-death presumption), "85 C.", titles without
# a closing full stop ("85-B. Presumptions as to electronic records and
# [electronic Signatures]"), and titles carrying an editorial remark inside
# the bold tag ("47A. Opinion as to [Electronic signature] [Inserted by Act
# 21 of 2000 ...]"). Found when a live answer's correct citation to "section
# 113-B" of the Evidence Act was stripped as nonexistent.
_RUN_IN = re.compile(r"^(\d+)\s?-?\s?([A-Za-z])\.\s*(.{4,160}?)\s*\.?\s*$", re.DOTALL)
_EDITORIAL_TAIL = re.compile(r"\s*\[+\s*(?:Inserted|Substituted|Omitted|Renumbered)\s+by\b.*$",
                             re.IGNORECASE | re.DOTALL)


def _embedded_headings(html: str):
    """(key_number, number_pattern, title) for every bold run-in heading."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for b in soup.find_all("b", class_="akn-b"):
        m = _RUN_IN.match(_clean(b.get_text(" ")))
        if not m:
            continue
        digits, letter, title = m.group(1), m.group(2).upper(), m.group(3)
        title = _EDITORIAL_TAIL.sub("", title).strip().rstrip(".").strip()
        if len(title) < 4:
            continue
        num_pattern = rf"{digits}\s?-?\s?{letter}"
        out.append((digits + letter, num_pattern, title))
    return out


def _split_embedded_subsections(sections: List[ScrapedSection], path: str,
                                source_url: str) -> List[ScrapedSection]:
    """
    A FIFTH markup variant, distinct from the akn-chapter case: an inserted
    sub-section (65A, 65B, ...) is rendered as a bolded run-in heading INSIDE
    its parent section's own flowing prose --
    `<span class="akn-p"><b class="akn-b">65A. Title.</b></span>` -- with no
    section-level tag boundary at all to hook into. IEA section 65 absorbed
    65A, 65B, 67A, 90A and 114A this way; IEA section 45 absorbed 45A. This
    matters more than most such defects: IEA section 65B is one of this
    project's own most-cited example provisions (the recodification mapping
    to BSA 63, the quotation-check discussion), and it was silently ABSENT as
    its own citable key throughout every measurement taken before this fix --
    found only while composing real generated answers for the generation
    experiment and finding "IEA§65B" had no entry in the database it should
    obviously be in.

    Headings are read from the raw HTML (the bold tag marks where one section
    ends and the embedded one begins, a boundary plain-text flattening loses),
    then located in the already-parsed body text by number AND title, so an
    ordinary cross-reference ("provisions of section 65B.") is never taken
    for the heading. The search runs over body text from which state/UT
    amendment tables have already been removed (is_state_amendment_table):
    a heading that exists only inside a state's amendment -- Uttar Pradesh's
    own s.90-A, West Bengal's s.78-A -- therefore finds nothing to split and
    never becomes a central-Act section, while the central s.90-A, which
    follows the UP table as plain text, is found and split out.
    """
    with open(path, "rb") as f:
        html = f.read().decode("utf-8", errors="ignore")
    embedded = _embedded_headings(html)
    if not embedded:
        return sections

    by_num = {s.number: s for s in sections}
    result = list(sections)
    for num, num_pattern, title in embedded:
        if num in by_num:
            continue  # already its own section; nothing to split out
        marker = re.compile(rf"\[?\s*\b{num_pattern}\.\s*{re.escape(title)}\s*\.?")
        for host in list(result):
            m = marker.search(host.text)
            if m is None:
                continue
            new_body = _dedup_halves(_clean(host.text[m.end():]))
            if not new_body:
                break
            host_new_text = _dedup_halves(_clean(host.text[:m.start()]))
            result[result.index(host)] = ScrapedSection(
                number=host.number, marginal_note=host.marginal_note,
                text=host_new_text, source_url=host.source_url)
            result.append(ScrapedSection(number=num, marginal_note=_clean(title),
                                         text=new_body, source_url=source_url))
            by_num[num] = result[-1]
            break
    return result


_NUM_ONLY = re.compile(r"^(\d+)")


def numeric_gaps(sections: List[ScrapedSection]) -> List[int]:
    """
    Integer section numbers missing from the observed range.

    A gap is reported as a FINDING, not silently absorbed: it usually means
    the source is missing a real, repealed/omitted provision -- most gaps
    below correspond to historically repealed sections, e.g. ICA 76-123 was
    the Sale of Goods chapter, repealed 1930. It can also mean the PARSER
    still has a defect: BNS section 318 ("Cheating") once appeared as a gap
    here and was reported as a confirmed source omission, until the true
    cause -- a markup variant this parser didn't yet handle, silently merging
    318's content onto section 317 -- was found and fixed (see
    test_bns_318_cheating_is_present_and_distinct_from_317 in
    tests/test_real_corpus.py). A gap is therefore a prompt to investigate,
    not a conclusion in itself. Lettered/inserted sections (66A, 420A) are not
    checked -- only the base numeric sequence is expected to be contiguous.
    """
    nums = sorted({int(m.group(1)) for s in sections
                  if (m := _NUM_ONLY.match(s.number))})
    if not nums:
        return []
    full = set(range(nums[0], nums[-1] + 1))
    return sorted(full - set(nums))


def scrape_all() -> dict:
    """Returns {act_code: [ScrapedSection, ...]} for every cached HTML source."""
    result = {}
    for name, fname, act, url, _word in ACT_SOURCES:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            continue
        result[act] = parse_akoma_html(path, url)
    return result


if __name__ == "__main__":
    data = scrape_all()
    for act, secs in data.items():
        print(f"{act:6} {len(secs):5} sections   e.g. {secs[0].number if secs else '-'}"
              f" [{(secs[0].marginal_note[:40] if secs else '')}]")
