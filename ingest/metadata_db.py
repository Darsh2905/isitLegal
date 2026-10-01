"""
Metadata database: the authority the citation verifier checks against.

Holds one row per statutory provision plus a bidirectional recodification
mapping. The verifier consults this to answer three questions:
  1. Does this cited provision exist at all?
  2. Is it currently in force, repealed, or otherwise invalid?
  3. If superseded, what is the corresponding current provision?
"""

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Provision:
    key: str
    act: str
    section: str
    marginal_note: str
    text: str
    repealed_by: Optional[str]
    verbatim: bool
    status_note: Optional[str] = None

    @property
    def in_force(self) -> bool:
        return self.repealed_by is None and self.status_note is None


class MetadataDB:
    def __init__(self, path: str = ":memory:"):
        # check_same_thread=False: app.py caches the whole MetadataDB instance
        # with @st.cache_resource, which exists specifically to share one
        # object (a DB connection, a loaded model) across every rerun and
        # session -- but Streamlit's script runner can execute a rerun on a
        # different thread than the one that created the cached object, and
        # sqlite3 refuses cross-thread use of a connection by default
        # (sqlite3.ProgrammingError). Safe to relax here: every write happens
        # once, synchronously, during construction in ingest/india_code.py or
        # data/seed_corpus.py, before the instance is ever cached or read
        # concurrently -- nothing after that point writes to this connection,
        # only SELECTs via all_provisions()/get()/exists()/successor().
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create()

    def _create(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS provisions (
                key           TEXT PRIMARY KEY,
                act           TEXT NOT NULL,
                section       TEXT NOT NULL,
                marginal_note TEXT,
                text          TEXT,
                repealed_by   TEXT,
                verbatim      INTEGER NOT NULL DEFAULT 0,
                status_note   TEXT
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS mapping (
                old_key  TEXT NOT NULL,
                new_key  TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (old_key, new_key)
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_act ON provisions(act)")
        self.conn.commit()

    # ---------------- ingestion ----------------
    def load_seed(self, sections, mapping, status_notes=None):
        status_notes = status_notes or {}
        c = self.conn.cursor()
        for act, sec, note, text, repealed_by, verbatim in sections:
            key = f"{act}§{sec}"
            c.execute(
                "INSERT OR REPLACE INTO provisions VALUES (?,?,?,?,?,?,?,?)",
                (key, act, sec, note, text, repealed_by,
                 1 if verbatim else 0, status_notes.get(key)),
            )
        for old, new, verified in mapping:
            c.execute("INSERT OR REPLACE INTO mapping VALUES (?,?,?)",
                      (old, new, 1 if verified else 0))
        self.conn.commit()

    # ---------------- lookup ----------------
    def get(self, key: str) -> Optional[Provision]:
        r = self.conn.execute(
            "SELECT * FROM provisions WHERE key = ?", (key,)).fetchone()
        if r is None:
            return None
        return Provision(
            key=r["key"], act=r["act"], section=r["section"],
            marginal_note=r["marginal_note"], text=r["text"],
            repealed_by=r["repealed_by"], verbatim=bool(r["verbatim"]),
            status_note=r["status_note"],
        )

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    _SUBSECTION_LABEL = re.compile(r"\(([0-9A-Za-z]+)\)")

    def resolve(self, key: str) -> Optional[Provision]:
        """
        Exact lookup, falling back from a sub-section pinpoint to its section.

        The real corpus indexes one row per SECTION (Section IV-A), so every
        correct pinpoint citation -- "Article 19(1)(a)", "Section 41(1)(b)
        CrPC", "Section 103(1) BNS" -- missed exact lookup and V2 stripped it
        as a nonexistent provision. The fallback is not a blanket acceptance
        of any sub-section of a real section: each cited label must occur in
        the parent section's own text AS A LABEL, nested inside the one before
        it -- see _labels_present. "IPC§302(7)" (IPC 302 has no sub-sections)
        and "CONST§19(2)(a)" (clause (2) of Art. 19 has no sub-clause (a); it
        only MENTIONS "sub-clause (a) of clause (1)") both still fail. This is
        evidence from the statute's own text, but weaker than an exact row:
        it confirms the pinpoint exists, not that the claim matches it.
        """
        prov = self.get(key)
        if prov is not None or "§" not in key:
            return prov
        act, section = key.split("§", 1)
        if "(" not in section:
            return None
        base = section.split("(", 1)[0]
        labels = self._SUBSECTION_LABEL.findall(section[len(base):])
        parent = self.get(f"{act}§{base}")
        if parent is None or not labels:
            return None
        return parent if _labels_present(parent.text or "", labels) else None

    def resolve_key(self, key: str) -> Optional[str]:
        prov = self.resolve(key)
        return prov.key if prov else None

    def successor(self, key: str) -> List[str]:
        """Current provision(s) corresponding to a superseded one."""
        return [r["new_key"] for r in self.conn.execute(
            "SELECT new_key FROM mapping WHERE old_key = ?", (key,))]

    def predecessor(self, key: str) -> List[str]:
        """Superseded provision(s) corresponding to a current one."""
        return [r["old_key"] for r in self.conn.execute(
            "SELECT old_key FROM mapping WHERE new_key = ?", (key,))]

    def is_mapping_verified(self, old_key: str, new_key: str) -> bool:
        r = self.conn.execute(
            "SELECT verified FROM mapping WHERE old_key=? AND new_key=?",
            (old_key, new_key)).fetchone()
        return bool(r["verified"]) if r else False

    def all_provisions(self) -> List[Provision]:
        # One query, not one per row: this is called on every retriever /
        # screen / gate construction and (formerly) on every pipeline query.
        # ORDER BY key reproduces the order the old per-key query returned
        # (a PRIMARY KEY index scan): retrievers break score ties by corpus
        # order, so changing it would silently change reported rankings.
        return [Provision(key=r["key"], act=r["act"], section=r["section"],
                          marginal_note=r["marginal_note"], text=r["text"],
                          repealed_by=r["repealed_by"],
                          verbatim=bool(r["verbatim"]),
                          status_note=r["status_note"])
                for r in self.conn.execute(
                    "SELECT * FROM provisions ORDER BY key")]

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) n FROM provisions").fetchone()["n"]


# A parenthesised label is a sub-section heading only when it is not a
# cross-reference: "(a)" in "sub-clause (a) of clause (1)" names another
# provision, it does not start one. Checked by the word immediately before it.
_REFERENCE_WORDS = {
    "clause", "sub-clause", "subclause", "section", "sub-section", "subsection",
    "sections", "clauses", "article", "articles", "paragraph", "sub-paragraph",
    "item", "items", "entry", "rule", "rules", "and", "or", "of", "to", "in",
    "under", "with", "by", "from",
}
_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi",
          "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]


def _label_positions(text: str, lab: str, lo: int, hi: int):
    needle = f"({lab})"
    pos = text.find(needle, lo, hi)
    while pos >= 0:
        window = text[max(0, pos - 40):pos]
        before = re.search(r"([a-z][a-z\-]*)\W*$", window)
        word = before.group(1) if before else None
        if word in ("and", "or"):
            # "(f) ...; or (g) ..." is a list item; "(a) and (b)" -- the
            # conjunction preceded by another label's ")" -- is a reference.
            prior = window[:before.start(1)].rstrip()
            is_reference = prior.endswith(")")
        else:
            is_reference = word in _REFERENCE_WORDS
        if not is_reference:
            yield pos
        pos = text.find(needle, pos + 1, hi)


def _next_siblings(lab: str):
    """Labels that would start the NEXT sub-section at the same level."""
    out = []
    if lab.isdigit():
        out.append(str(int(lab) + 1))
    if lab in _ROMAN and _ROMAN.index(lab) + 1 < len(_ROMAN):
        out.append(_ROMAN[_ROMAN.index(lab) + 1])
    if len(lab) == 1 and lab.isalpha() and lab < "z":
        out.append(chr(ord(lab) + 1))
    return out


def _labels_present(text: str, labels) -> bool:
    """Each label occurs as a real label (not a cross-reference), each inside
    the span of the one before it -- the span ending where the parent's next
    sibling begins."""
    text = text.lower()
    lo, hi = 0, len(text)
    for lab in (l.lower() for l in labels):
        pos = next(_label_positions(text, lab, lo, hi), None)
        if pos is None:
            return False
        ends = [p for s in _next_siblings(lab)
                for p in [next(_label_positions(text, s, pos + 1, hi), None)]
                if p is not None]
        lo, hi = pos + 1, min(ends, default=hi)
    return True


def build_seed_db(path: str = ":memory:") -> MetadataDB:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.seed_corpus import SECTIONS, MAPPING, STATUS_NOTES
    db = MetadataDB(path)
    db.load_seed(SECTIONS, MAPPING, STATUS_NOTES)
    return db
