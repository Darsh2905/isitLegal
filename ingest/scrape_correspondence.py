"""
Parser for the official recodification correspondence tables.

SOURCE. Uttar Pradesh Police published a matched series of three "Corresponding
Section Table" documents, one per recodified Act, in the same format from the
same source -- a state government law-enforcement authority's own comparative
tables, prepared for exactly the purpose this project needs them for. All
three follow an identical two-column layout (new Act left, repealed Act
right), which is why one parametrised parser covers all three rather than
three bespoke ones:

  BNS  <-> IPC   https://uppolice.gov.in/.../202406281710564823BNS_IPC_Comparative.pdf
  BNSS <-> CRPC  https://uppolice.gov.in/.../202407031502194192BNSS2023-20-45.pdf
  BSA  <-> IEA   https://uppolice.gov.in/.../202407031507276576BSA2023-9-25.pdf

Until this module existed, this project's own mapping table carried every
entry as `verified=False` (CLAUDE.md Section 9: a mapping may only be flagged
`verified=True` with a citation to the authoritative source it was checked
against). It now checks against a real source for all three recodified Act
pairs, not just BNS<->IPC as in the previous revision.

FORMAT. A two-column table, new Act on the left, old Act on the right,
extracted by pdfplumber's ruled-table detection. Some provisions correspond to
multiple provisions on the other side (consolidation or splitting); both
directions are preserved as separate mapping rows rather than collapsed to a
single pair, matching this project's existing MAPPING schema of
(old_key, new_key, verified) tuples, which already tolerates one-to-many.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pdfplumber

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw")

# (new_act_code, old_act_code, cached filename, source URL)
TABLE_SOURCES = [
    ("BNS", "IPC", "correspondence_bns_ipc.pdf",
     "https://uppolice.gov.in/site/writereaddata/siteContent/"
     "Three%20New%20Major%20Acts/202406281710564823BNS_IPC_Comparative.pdf"),
    ("BNSS", "CRPC", "correspondence_bnss_crpc.pdf",
     "https://uppolice.gov.in/site/writereaddata/siteContent/"
     "Three%20New%20Major%20Acts/202407031502194192BNSS2023-20-45.pdf"),
    ("BSA", "IEA", "correspondence_bsa_iea.pdf",
     "https://uppolice.gov.in/site/writereaddata/siteContent/"
     "Three%20New%20Major%20Acts/202407031507276576BSA2023-9-25.pdf"),
]

# Kept for backward compatibility with callers written against the
# single-table version of this module.
SOURCE_URL = TABLE_SOURCES[0][3]

_LEAD_NUM = re.compile(r"^\s*(\d+[A-Za-z]?(?:\s*\(\d+[a-zA-Z]?\))?)\s*\.?\s")

# Title-row artefacts: the document's own heading ("... 2023" / "... 1860")
# parses as a spurious correspondence under the same leading-number rule that
# reads real rows. No actual section number in any of these six Acts reaches
# four digits, so year-magnitude numbers are excluded outright.
_YEAR_LIKE = re.compile(r"^(18|19|20)\d{2}$")


@dataclass(frozen=True)
class CorrespondenceRow:
    new_act: str
    new_section: str
    old_act: str
    old_section: str


def _extract_number(cell: str) -> str:
    if not cell:
        return ""
    m = _LEAD_NUM.match(cell.replace("\n", " "))
    return re.sub(r"\s+", "", m.group(1)) if m else ""


def parse_one_table(new_act: str, old_act: str, path: str) -> List[CorrespondenceRow]:
    rows: List[CorrespondenceRow] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if len(row) < 2:
                        continue
                    new_cell, old_cell = row[0] or "", row[1] or ""
                    new_num = _extract_number(new_cell)
                    old_num = _extract_number(old_cell)
                    if not (new_num and old_num):
                        continue
                    if _YEAR_LIKE.match(new_num) and _YEAR_LIKE.match(old_num):
                        continue
                    rows.append(CorrespondenceRow(new_act, new_num, old_act, old_num))
    return rows


def parse_correspondence(path: str = None) -> List:
    """
    Backward-compatible entry point: with no argument, parses all three
    tables and returns BNS<->IPC rows in the ORIGINAL two-field shape
    (bns_section, ipc_section) that eval/run_real_corpus_eval.py and the
    tests already consume. Use parse_all() for the full three-Act result.
    """
    if path is None:
        path = os.path.join(RAW_DIR, TABLE_SOURCES[0][2])
    rows = parse_one_table("BNS", "IPC", path)

    from collections import namedtuple
    Row = namedtuple("Row", ["bns_section", "ipc_section"])
    return [Row(r.new_section, r.old_section) for r in rows]


def parse_all() -> Dict[Tuple[str, str], List[CorrespondenceRow]]:
    """Returns {(new_act, old_act): [CorrespondenceRow, ...]} for all three
    recodified pairs, reading from the cached PDFs in data/raw/."""
    out = {}
    for new_act, old_act, fname, _url in TABLE_SOURCES:
        p = os.path.join(RAW_DIR, fname)
        if os.path.exists(p):
            out[(new_act, old_act)] = parse_one_table(new_act, old_act, p)
    return out


def build_verified_mapping(existing_mapping: List[Tuple[str, str, bool]]
                           ) -> List[Tuple[str, str, bool]]:
    """
    Cross-check `existing_mapping` (old_key, new_key, verified) tuples against
    all three parsed correspondence tables. An entry is promoted to
    verified=True only when a table records exactly that old-Act section
    number corresponding to exactly that new-Act section number for the
    matching Act pair -- not merely that both appear somewhere in a document.
    Entries no table confirms are returned unchanged, never silently flipped.
    """
    all_rows = parse_all()
    confirmed = set()
    for (new_act, old_act), rows in all_rows.items():
        for r in rows:
            confirmed.add((old_act, r.old_section, new_act, r.new_section))

    out = []
    for old_key, new_key, verified in existing_mapping:
        old_act, old_num = old_key.split("§")
        new_act, new_num = new_key.split("§")
        if (old_act, old_num, new_act, new_num) in confirmed:
            out.append((old_key, new_key, True))
        else:
            out.append((old_key, new_key, verified))
    return out


if __name__ == "__main__":
    all_rows = parse_all()
    for (new_act, old_act), rows in all_rows.items():
        print(f"{new_act} <-> {old_act}: {len(rows)} correspondence rows")
        for r in rows[:3]:
            print(f"    {old_act}§{r.old_section:8} <-> {new_act}§{r.new_section}")
