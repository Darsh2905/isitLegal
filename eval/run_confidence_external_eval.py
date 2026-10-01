"""
Out-of-sample check: the ALREADY-DEPLOYED, already-calibrated real-corpus
confidence gate (retrieval/confidence.py, tau tuned by
eval/run_confidence_real_eval.py) against eight genuinely externally-sourced
unanswerable queries (eval/external_unanswerable_qa.py) -- real page titles
from a public legal Q&A platform, not authored by this project.

This is deliberately NOT a retuning exercise. No threshold here is fit to
these eight queries; the gate is used exactly as deployed, and the only
question asked is whether it generalises to questions it has never seen and
did not shape. That is the honest test Limitation Seven's "author-written,
not adversarially sourced" gap calls for -- see
eval/external_unanswerable_qa.py's docstring for what this does and does not
establish (a small set, sourced from one platform, not a large adversarial
corpus).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.confidence import ConfidenceGate
from eval.external_unanswerable_qa import EXTERNAL_UNANSWERABLE

REAL_CORPUS_TAU = 1.077  # the deployed value -- not retuned here


def report():
    db = build_real_db()
    # The gate's signal is calibrated on UNSTEMMED BM25: stemming, though it
    # improves retrieval, raises lexical overlap for out-of-scope questions too
    # and cut held-out correct abstention 1.000 -> 0.625 (retrieval/text.py).
    bm25 = BM25Retriever(db, stem=False)
    gate = ConfidenceGate(db, bm25, tau=REAL_CORPUS_TAU)

    print("OUT-OF-SAMPLE CHECK: EXTERNALLY-SOURCED UNANSWERABLE QUERIES")
    print("=" * 78)
    print(f"Deployed gate, tau={REAL_CORPUS_TAU} -- NOT retuned to these queries.")
    print(f"{len(EXTERNAL_UNANSWERABLE)} real question titles from law4u.in, a public")
    print("legal Q&A platform. See eval/external_unanswerable_qa.py for source URLs.")
    print("-" * 78)

    wrongly_answered = 0
    rows = []
    for q, url in EXTERNAL_UNANSWERABLE:
        conf = gate.confidence(q)
        answerable = conf.score >= REAL_CORPUS_TAU
        wrongly_answered += answerable
        rows.append({"query": q, "url": url, "score": conf.score,
                    "answerable": answerable})
        print(f"  {'ANSWER' if answerable else 'abstain':8}{conf.score:>7.3f}  {q[:60]}")

    n = len(EXTERNAL_UNANSWERABLE)
    correct_abstention = 1 - wrongly_answered / n
    print("-" * 78)
    print(f"Wrongly answered: {wrongly_answered}/{n}   "
          f"correct abstention: {correct_abstention:.3f}")

    return {"n": n, "wrongly_answered": wrongly_answered,
            "correct_abstention": correct_abstention, "rows": rows,
            "tau": REAL_CORPUS_TAU}


if __name__ == "__main__":
    report()
