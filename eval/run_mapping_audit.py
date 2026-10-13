"""
Mapping audit: does a second, independent source agree with the IPC<->BNS
correspondence the verifier relies on?

The mapping table is built from correspondence tables published by the Uttar
Pradesh Police (ingest/scrape_correspondence.py). Entries are flagged
verified=True because they come from a published table -- which records their
source, not that the source is right. A 2026-10 peer review pointed out that
"official" overstated a state police department's document. The honest
description is "a published government correspondence table", and the
useful measurement is agreement with an independent one.

SECOND SOURCE. "Correspondence Table and Comparison Summary of the Bharatiya
Nyaya Sanhita, 2023 to the Indian Penal Code, 1860", A. K. Yadav, Director,
Central Academy for Police Training (CAPT), Bhopal -- a central police
training institution, independent of the UP Police tables -- hosted by the
Kerala Prisons and Correctional Services department, in data/official/. It
covers BNS<->IPC only; no independent table for BNSS<->CrPC or BSA<->IEA was
obtainable, so those pairs remain single-source and the paper says so.

COMPARISON, at section level (sub-section pinpoints dropped on both sides):
for every IPC section either source maps, do the two sources name the same
BNS section(s)? Reported as exact agreement, partial overlap, and
disagreement, with every disagreement listed for inspection.
"""
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "official", "capt_bhopal_bns_ipc.pdf")
_BNS = re.compile(r"^\s*(\d{1,3})\s*(?:\(\s*\d+\s*\))?(?:\s*\([a-z]+\))*\s*$")
_IPC = re.compile(r"\b(\d{1,3}[A-Z]{0,2})\b")


def parse_capt():
    """IPC section -> set of BNS sections, from the CAPT Bhopal table."""
    import pymupdf
    doc = pymupdf.open(PDF)
    pairs = defaultdict(set)
    rows = 0
    for page in doc:
        for t in page.find_tables().tables:
            for r in t.extract():
                cells = [(c or "").replace("\n", " ").strip() for c in r]
                cells = [c for c in cells if c]
                if len(cells) < 3:
                    continue
                m = _BNS.match(cells[0])
                if not m:
                    continue
                ipc_cell = cells[2]
                if re.match(r"(?i)\s*new\b", ipc_cell):
                    continue
                rows += 1
                for s in _IPC.findall(ipc_cell):
                    pairs[s].add(m.group(1))
    return pairs, rows


def report():
    from ingest.india_code import build_real_db
    db = build_real_db()
    capt, rows = parse_capt()
    ours = defaultdict(set)
    for p in db.all_provisions():
        if p.act != "IPC":
            continue
        for s in db.successor(p.key):
            if s.startswith("BNS§"):
                ours[p.section].add(s.split("§", 1)[1].split("(", 1)[0])
    both = sorted(set(capt) & set(ours), key=lambda s: (int(re.match(r"\d+", s).group()), s))
    exact = [s for s in both if capt[s] == ours[s]]
    overlap = [s for s in both if capt[s] != ours[s] and capt[s] & ours[s]]
    disagree = [s for s in both if not (capt[s] & ours[s])]
    print("MAPPING AUDIT: UP Police IPC->BNS table vs CAPT Bhopal table")
    print("=" * 78)
    print(f"  CAPT rows parsed: {rows};  IPC sections mapped -- CAPT {len(capt)}, "
          f"ours {len(ours)}, both {len(both)}")
    print(f"  exact agreement : {len(exact)}/{len(both)} ({len(exact)/len(both):.3f})")
    print(f"  partial overlap : {len(overlap)}")
    print(f"  disagreement    : {len(disagree)}")
    for s in overlap:
        print(f"     overlap   IPC {s:6} ours {sorted(ours[s])}  CAPT {sorted(capt[s])}")
    for s in disagree:
        print(f"     DISAGREE  IPC {s:6} ours {sorted(ours[s])}  CAPT {sorted(capt[s])}")
    only_ours = sorted(set(ours) - set(capt))
    only_capt = sorted(set(capt) - set(ours))
    print(f"  mapped only in ours: {len(only_ours)};  only in CAPT: {len(only_capt)}")
    return {"both": len(both), "exact": len(exact), "overlap": len(overlap),
            "disagree": len(disagree), "disagreements":
            [(s, sorted(ours[s]), sorted(capt[s])) for s in disagree],
            "overlaps": [(s, sorted(ours[s]), sorted(capt[s])) for s in overlap],
            "only_ours": len(only_ours), "only_capt": len(only_capt)}


if __name__ == "__main__":
    report()
