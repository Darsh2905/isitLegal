import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.normalize import normalise, extract_citations
from eval.normalizer_testset import CASES, NEGATIVES


def report():
    ok = 0; fails = []
    for surface, expected in CASES:
        got = normalise(surface)
        if got == expected: ok += 1
        else: fails.append((surface, expected, got))

    fp = 0; fp_ex = []
    for s in NEGATIVES:
        c = extract_citations(s)
        if c:
            fp += 1; fp_ex.append((s, [x.key for x in c]))

    print(f"Positive cases : {ok}/{len(CASES)}  ({100*ok/len(CASES):.1f}%)")
    print(f"False positives: {fp}/{len(NEGATIVES)}")
    if fails:
        print("\n-- FAILURES --")
        for s,e,g in fails: print(f"  {s!r}\n     expected {e}  got {g}")
    if fp_ex:
        print("\n-- FALSE POSITIVES --")
        for s,k in fp_ex: print(f"  {s!r} -> {k}")

    return {
        "positives_correct": ok,
        "positives_total": len(CASES),
        "positive_accuracy": ok / len(CASES),
        "false_positives": fp,
        "negatives_total": len(NEGATIVES),
        "failures": fails,
        "false_positive_examples": fp_ex,
    }


if __name__ == "__main__":
    report()
