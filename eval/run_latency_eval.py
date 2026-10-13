"""
Verification latency on real generated answers.

The paper compares the verifier's cost with the 12.8 s that graph traversal
and adversarial generation add in Bindra et al. The earlier figure (1.9 ms
median) was measured ad hoc and no script reproduced it, against CLAUDE.md
invariant 6; this is that script. Measured after the fail-closed revision,
which added bare-reference placement, case-citation resolution and the
residual scan to every sentence.

Wall-clock time of CitationVerifier.verify (repair and V5 on, as deployed)
per answer, best of REPEATS runs per answer to reduce scheduler noise, over
every answer in both live rounds, with the parser's per-sentence memo cleared
before each run so every timing is cold. Index construction (corpus load, case
index) is a one-off start-up cost and is excluded, as it is for the
retrieval figures it is compared with.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPEATS = 5


def answers():
    from eval.live_generation_answers import LIVE_ANSWERABLE
    out = [(f["raw_text"], f["chunk_keys"]) for f in LIVE_ANSWERABLE.values()
           if f["outcome"] == "generated"]
    try:
        from eval.run_live_round2_eval import load
        data = load()
        for a in data["answers"].values():
            if a["text"] is None:
                continue
            keys = (data["retrieval"][a["qid"]]["chunk_keys"]
                    if a["condition"] == "grounded" else [])
            out.append((a["text"], keys))
    except FileNotFoundError:
        pass
    return out


def _cold():
    from ingest import normalize
    normalize._extract_citations_cached.cache_clear()
    normalize._unresolved_cached.cache_clear()


def report():
    from ingest.india_code import build_real_db
    from ingest.case_citations import build_case_index
    from verify.verifier import CitationVerifier, Chunk
    db = build_real_db()
    ver = CitationVerifier(db, repair=True, support=True,
                           cases=build_case_index())
    times = []
    for text, keys in answers():
        chunks = [Chunk(i, k, db.get(k).text, db.get(k).verbatim)
                  for i, k in enumerate(keys, 1)]
        best = float("inf")
        for _ in range(REPEATS):
            # Cold every time: the parser memoises per sentence, and a warm
            # cache would time the lookup, not the verification.
            _cold()
            t = time.perf_counter()
            ver.verify(text, chunks)
            best = min(best, time.perf_counter() - t)
        times.append(best * 1000)
    times.sort()
    med = statistics.median(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    print("VERIFICATION LATENCY (ms per answer, repair + V5 on)")
    print("=" * 78)
    print(f"  answers {len(times)}   median {med:.1f}   p95 {p95:.1f}   "
          f"max {times[-1]:.1f}")
    return {"n": len(times), "median_ms": med, "p95_ms": p95,
            "max_ms": times[-1]}


if __name__ == "__main__":
    report()
