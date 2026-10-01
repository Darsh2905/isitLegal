"""
Diagnostic: does retrieval prefer CURRENT law over its repealed predecessor?

Repealed and successor provisions share a marginal note almost verbatim
("Punishment for murder" appears in both IPC 302 and BNS 103), so a purely
lexical retriever has no signal distinguishing them. This is precisely the
failure mode that motivates contribution C2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.bm25_baseline import BM25Retriever
from eval.gold_qa import GOLD


def report():
    # Toy-corpus measurement, pinned to the original unstemmed retriever so
    # this long-reported diagnostic is unchanged; the real-corpus version
    # is measured separately below.
    db = build_seed_db(); r = BM25Retriever(db, stem=False)
    pairs = [(q, rel, inf) for _, q, rel, inf in GOLD
             if inf and any(db.get(k) and not db.get(k).in_force for k in rel)]

    correct = 0; rows = []
    for q, rel, inf in pairs:
        ranked = [k for k, _ in r.search(q, 30)]
        pos_inf = ranked.index(inf) if inf in ranked else 99
        repealed = [k for k in rel if k != inf]
        pos_rep = min((ranked.index(k) for k in repealed if k in ranked), default=99)
        ok = pos_inf < pos_rep
        correct += ok
        rows.append((q[:42], inf, pos_inf+1, repealed[0] if repealed else "-",
                     pos_rep+1, ok))

    print("IN-FORCE DISCRIMINATION (BM25 only, no verifier)")
    print("="*78)
    print(f"{'query':44}{'in-force':>10}{'rk':>4}{'repealed':>11}{'rk':>4}  ok")
    print("-"*78)
    for q, inf, pi, rep, pr, ok in rows:
        print(f"{q:44}{inf:>10}{pi:>4}{rep:>11}{pr:>4}  {'Y' if ok else 'N'}")
    print("-"*78)
    print(f"In-force ranked above repealed: {correct}/{len(pairs)} "
          f"= {correct/len(pairs):.3f}")
    print("\nWith the verifier's recodification annotation, a repealed hit is")
    print("labelled and its successor supplied, so the user is not misled")
    print("regardless of rank. Measured separately below.")

    from verify.verifier import CitationVerifier, Chunk
    ver = CitationVerifier(db)
    ann = 0
    for q, rel, inf in pairs:
        repealed = [k for k in rel if k != inf][0]
        p = db.get(repealed)
        ch = [Chunk(1, repealed, p.text, p.verbatim)]
        sec, num = repealed.split("§")
        out, rep_ = ver.verify(f"This is governed by Section {num} {sec} [1].", ch)
        if rep_.annotations and inf in " ".join(rep_.annotations):
            ann += 1
    print(f"Repealed citation correctly annotated with successor: "
          f"{ann}/{len(pairs)} = {ann/len(pairs):.3f}")

    real = _real_corpus()

    return {"real": real,
            "n_pairs": len(pairs),
            "inforce_above_repealed": correct,
            "inforce_above_repealed_rate": correct/len(pairs),
            "annotated": ann,
            "annotated_rate": ann/len(pairs),
            "rows": rows}


def _real_corpus():
    """
    The same diagnostic on the real 3,263-provision corpus, which the toy
    measurement above could only stand in for, with the full verified
    correspondence mapping and the in-force preference step
    (retrieval/inforce.py) that mapping makes possible.
    """
    from ingest.india_code import build_real_db
    from retrieval.inforce import prefer_in_force
    db = build_real_db()
    pairs = [(q, rel, inf) for _, q, rel, inf in GOLD
             if inf and any(db.get(k) and not db.get(k).in_force for k in rel)]

    def rate(retriever, prefer):
        ok = 0
        for q, rel, inf in pairs:
            ranked = retriever.search(q, 30)
            if prefer:
                ranked = prefer_in_force(ranked, db)
            keys = [k for k, _ in ranked]
            pi = keys.index(inf) if inf in keys else 99
            pr = min((keys.index(k) for k in rel if k != inf and k in keys), default=99)
            ok += pi < pr
        return ok

    arms = [("BM25 unstemmed", BM25Retriever(db, stem=False), False),
            ("BM25 stemmed", BM25Retriever(db, stem=True), False),
            ("BM25 stemmed + in-force preference", BM25Retriever(db, stem=True), True)]
    print("\nIN-FORCE DISCRIMINATION ON THE REAL CORPUS")
    print("=" * 78)
    out = {"n_pairs": len(pairs)}
    for label, retr, prefer in arms:
        ok = rate(retr, prefer)
        out[label] = ok
        print(f"  {label:38} in-force above repealed: {ok}/{len(pairs)} "
              f"= {ok/len(pairs):.3f}")
    print("  In-force preference only reorders what retrieval found; the pairs")
    print("  it cannot fix are those whose in-force provision was not retrieved")
    print("  in the top 30 at all.")

    from verify.verifier import CitationVerifier, Chunk
    ver = CitationVerifier(db)
    ann = 0
    for q, rel, inf in pairs:
        repealed = [k for k in rel if k != inf][0]
        p = db.get(repealed)
        sec, num = repealed.split("§")
        _, rep_ = ver.verify(f"This is governed by Section {num} {sec} [1].",
                             [Chunk(1, repealed, p.text, p.verbatim)])
        ann += bool(rep_.annotations and inf in " ".join(rep_.annotations))
    out["annotated"] = ann
    print(f"  Repealed citation annotated with its successor: {ann}/{len(pairs)}")
    return out


if __name__ == "__main__":
    report()
