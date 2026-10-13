"""
Provenance check: does the corpus reproduce the OFFICIAL statutory text?

The statutory corpus is scraped from Indian Kanoon, not an official source
(India Code refused automated access). A 2026-10 peer review asked that this
be stated plainly; it also invites a measurement, since the verifier's
quotation check (V4) is only as good as the text it checks against.

OFFICIAL TEXT. The Gazette of India (Extraordinary), Part II, Section 1,
25 December 2023, as published by the Ministry of Law and Justice
(Legislative Department) -- Nos. 53, 54, 55 -- for the three codes the
recodification turns on: the Bharatiya Nyaya Sanhita (Act 45 of 2023),
the Bharatiya Nagarik Suraksha Sanhita (Act 46) and the Bharatiya Sakshya
Adhiniyam (Act 47). Copies from PRS Legislative Research, in data/official/.
The repealed codes (IPC, CrPC, IEA) and the other Acts are NOT checked here:
no official copy was obtainable, so their provenance remains Indian Kanoon
alone, and the paper says so.

METHOD. Token 8-gram ("shingle") coverage: the fraction of a corpus
section's 8-token windows that occur anywhere in the official text. PDF
extraction interleaves the Gazette's marginal notes and page furniture with
the body text, which breaks a few windows per page even where the wording is
identical, so a contiguous-match test would understate agreement; shingle
coverage tolerates those breaks while still failing a section whose wording
differs. Every section below the threshold is listed for inspection rather
than summarised away.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OFFICIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "official")
FILES = {"BNS": "The_Bharatiya_Nyaya_Sanhita__2023.pdf",
         "BNSS": "The_Bharatiya_Nagarik_Suraksha_Sanhita__2023.pdf",
         "BSA": "The_Bharatiya_Sakshya_Adhiniyam__2023.pdf"}
SHINGLE = 8
THRESHOLD = 0.90
_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s):
    return _TOK.findall(s.lower())


def _official_shingles(act):
    import pymupdf
    doc = pymupdf.open(os.path.join(OFFICIAL, FILES[act]))
    toks = _tokens(" ".join(p.get_text() for p in doc))
    return {tuple(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def coverage(text, shingles):
    t = _tokens(text)
    if len(t) < SHINGLE:
        return None          # too short to test this way
    win = [tuple(t[i:i + SHINGLE]) for i in range(len(t) - SHINGLE + 1)]
    return sum(w in shingles for w in win) / len(win)


def report():
    from ingest.india_code import build_real_db
    db = build_real_db()
    print("PROVENANCE: corpus text vs the Gazette of India text (BNS, BNSS, BSA)")
    print("=" * 78)
    out = {}
    for act in FILES:
        sh = _official_shingles(act)
        provs = [p for p in db.all_provisions() if p.act == act]
        rows = []
        for p in provs:
            c = coverage(p.text or "", sh)
            rows.append((p.key, c))
        tested = [(k, c) for k, c in rows if c is not None]
        ok = [k for k, c in tested if c >= THRESHOLD]
        low = sorted([(c, k) for k, c in tested if c < THRESHOLD])
        mean = sum(c for _, c in tested) / len(tested)
        out[act] = {"sections": len(provs), "tested": len(tested),
                    "at_or_above": len(ok), "mean_coverage": mean,
                    "below": [(k, c) for c, k in low]}
        print(f"\n  {act}: {len(provs)} sections, {len(tested)} long enough to test; "
              f"{len(ok)} ({len(ok)/len(tested):.3f}) at >= {THRESHOLD:.2f} shingle "
              f"coverage; mean {mean:.3f}")
        for c, k in low:
            print(f"     below threshold: {k:10} {c:.3f}")
    tot_t = sum(v["tested"] for v in out.values())
    tot_ok = sum(v["at_or_above"] for v in out.values())
    print(f"\n  ALL THREE: {tot_ok}/{tot_t} sections ({tot_ok/tot_t:.3f}) "
          f"match the Gazette text at >= {THRESHOLD:.2f}")
    return {"acts": out, "tested": tot_t, "matching": tot_ok,
            "threshold": THRESHOLD}


if __name__ == "__main__":
    report()
