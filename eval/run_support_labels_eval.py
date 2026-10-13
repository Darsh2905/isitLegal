"""
SUPPORT, as distinct from RESOLUTION, on real generated output.

The verifier guarantees that every surviving citation RESOLVES -- the marker
points at a retrieved chunk, the provision or case exists, the quotation is
verbatim. It does not guarantee that the cited source SUPPORTS the claim; V5
screens for that, advisory only. A 2026-10 peer review observed that the
paper's "precision 1.0" was therefore a resolution figure presented under a
name (citation precision, "the fraction of cited sources that support the
claim") that means support. This script measures support directly.

SAMPLE. From the second live round (eval/live_round2_answers.json), after
verification: every surviving sentence that carries a citation (a marker
naming a retrieved chunk, or an explicit statutory citation that resolves) is
eligible. SAMPLE_PER_CELL sentences are drawn per (model, condition) with a
fixed seed, so the sample is reproducible and was fixed before any label was
written.

LABELS (eval/support_labels.py), one per sampled sentence, judged by reading
the sentence against the full text of every provision it cites:
  supported   -- the cited text states what the sentence claims;
  partial     -- the cited text supports part of the claim, or the claim
                 needs an inference the text does not make;
  unsupported -- the cited text does not support the claim (a real provision
                 attached to the wrong claim -- the failure V2 cannot see).
The labels are the authors' own, by one annotator, not legal experts: this is
the paper's weakest-sourced number and is reported as such.

REPORTED: support precision (supported / labelled, and supported + partial)
with Wilson intervals, per condition; and V5's agreement with the labels --
of the unsupported sentences, how many V5 flagged (its recall on real
output), and of the sentences V5 flagged, how many were unsupported.
"""
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_PER_CELL = 15
SEED = 2026
_MARK = re.compile(r"[\[【［](\d+)[\]】］]")


def eligible_sentences(rows, db):
    """(answer key, index, sentence, cited keys) for every surviving sentence
    that carries a resolvable citation."""
    from verify.verifier import _segment
    from ingest.normalize import extract_citations
    from eval.citation_audit import strip_annotations
    out = []
    for r in rows:
        text = strip_annotations(r["output"])
        if text.startswith("Insufficient verifiable authority"):
            continue
        chunk_keys = r.get("chunk_keys") or []
        for i, (sent, _b) in enumerate(_segment(text)):
            keys = []
            for n in _MARK.findall(sent):
                n = int(n)
                if 1 <= n <= len(chunk_keys):
                    keys.append(chunk_keys[n - 1])
            for c in extract_citations(sent):
                p = db.resolve(c.key)
                if p is not None:
                    keys.append(p.key)
            keys = list(dict.fromkeys(keys))
            if keys:
                out.append((r["key"], i, sent, keys))
    return out


def draw_sample():
    """The fixed sample: SAMPLE_PER_CELL per (model, condition)."""
    from ingest.india_code import build_real_db
    from eval.run_live_round2_eval import verify_all, load
    data = load()
    db = build_real_db()
    rows = verify_all(data)
    for r in rows:
        r["chunk_keys"] = (data["retrieval"][r["qid"]]["chunk_keys"]
                           if r["condition"] == "grounded" else [])
    by_cell = defaultdict(list)
    meta = {r["key"]: r for r in rows}
    for item in eligible_sentences(rows, db):
        m = meta[item[0]]
        by_cell[(m["model"], m["condition"])].append(item)
    rng = random.Random(SEED)
    sample = []
    for cell in sorted(by_cell):
        items = sorted(by_cell[cell], key=lambda t: (t[0], t[1]))
        pick = rng.sample(items, min(SAMPLE_PER_CELL, len(items)))
        for key, i, sent, keys in sorted(pick, key=lambda t: (t[0], t[1])):
            sample.append({"id": f"{key}#{i}", "model": cell[0],
                           "condition": cell[1], "sentence": sent,
                           "keys": keys})
    return sample, db


def v5_flags(sample, db):
    """V5's verdict on each sampled sentence (WEAK = flagged)."""
    from verify.support import SupportScreen
    # As deployed (pipeline.py): tau 0.20, no citation graph.
    screen = SupportScreen(db, 0.20)
    out = {}
    for s in sample:
        verdict, _best = screen.resolve_best(s["sentence"], s["keys"])
        out[s["id"]] = verdict == SupportScreen.WEAK
    return out


def report():
    from eval.stats import wilson
    from eval.support_labels import LABELS
    sample, db = draw_sample()
    flags = v5_flags(sample, db)
    print("SUPPORT ON REAL OUTPUT: author-labelled sample of surviving cited "
          "sentences")
    print("=" * 78)
    # A label counts only for the exact sentence it was given for.
    LABELS = {k: v for k, v in LABELS.items()
              if any(s["id"] == k and s["sentence"] == v[2] for s in sample)}
    missing = [s["id"] for s in sample if s["id"] not in LABELS]
    if missing:
        print(f"  {len(missing)} sampled sentences are not labelled yet")
    res = {}
    for cond in ("grounded", "ungrounded", "all"):
        ss = [s for s in sample if s["id"] in LABELS
              and (cond == "all" or s["condition"] == cond)]
        n = len(ss)
        if not n:
            continue
        lab = [LABELS[s["id"]][0] for s in ss]
        sup = lab.count("supported")
        part = lab.count("partial")
        uns = lab.count("unsupported")
        flagged = [s for s in ss if flags[s["id"]]]
        uns_ids = [s["id"] for s in ss if LABELS[s["id"]][0] == "unsupported"]
        v5_tp = sum(1 for s in flagged if LABELS[s["id"]][0] == "unsupported")
        res[cond] = {"n": n, "supported": sup, "partial": part,
                     "unsupported": uns,
                     "support_precision": sup / n,
                     "support_precision_ci": wilson(sup, n),
                     "support_or_partial": (sup + part) / n,
                     "support_or_partial_ci": wilson(sup + part, n),
                     "v5_flagged": len(flagged), "v5_true_flags": v5_tp,
                     "v5_recall_on_unsupported":
                         v5_tp / len(uns_ids) if uns_ids else None}
        lo, hi = wilson(sup, n)
        lo2, hi2 = wilson(sup + part, n)
        print(f"\n  {cond.upper()}: {n} labelled sentences")
        print(f"    supported            {sup:3}  -> support precision "
              f"{sup/n:.3f} (95% CI {lo:.2f}-{hi:.2f})")
        print(f"    partial              {part:3}  -> supported or partial "
              f"{(sup+part)/n:.3f} (95% CI {lo2:.2f}-{hi2:.2f})")
        print(f"    unsupported          {uns:3}")
        print(f"    V5 flagged {len(flagged)}; of those, unsupported {v5_tp}; "
              f"V5 caught {v5_tp}/{len(uns_ids)} unsupported")
    print("\n  every UNSUPPORTED sentence, with the annotator's note:")
    for s in sample:
        if s["id"] in LABELS and LABELS[s["id"]][0] == "unsupported":
            print(f"    [{s['condition'][:4]}] {s['sentence'][:90]!r}")
            print(f"           cites {s['keys']}  -- {LABELS[s['id']][1]}")
    return {"by_condition": res, "n_sample": len(sample),
            "n_missing": len(missing)}


if __name__ == "__main__":
    report()
