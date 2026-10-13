"""
The fail-closed revision's probe set (eval/verifier_failclosed_probes.py) on
the real corpus.

THE "BEFORE" COLUMN. Run with CODE_ROOT pointing at a checkout of the code as
it stood before the revision, to measure what that code did with the same
probes:

    git worktree add /tmp/before 4da3180
    ln -s "$PWD/data/raw" /tmp/before/data/raw
    rm -rf /tmp/before/data/raw_judgments
    ln -s "$PWD/data/raw_judgments" /tmp/before/data/raw_judgments
    CODE_ROOT=/tmp/before python3 eval/run_failclosed_eval.py

Only the verifier, normaliser and corpus come from CODE_ROOT; the probes are
always read from this checkout, so both runs score the same inputs.
"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_ROOT = os.environ.get("CODE_ROOT") or HERE
sys.path.insert(0, CODE_ROOT)

_spec = importlib.util.spec_from_file_location(
    "failclosed_probes", os.path.join(HERE, "eval", "verifier_failclosed_probes.py"))
_probes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probes)
PROBES = _probes.PROBES

ADVISORY = {"repealed_provision", "invalidated_provision", "weak_support",
            "support_via_reference", "marker_mismatch"}


def _verifier(db):
    from verify.verifier import CitationVerifier
    try:
        from ingest.case_citations import build_case_index
        return CitationVerifier(db, repair=True, support=True,
                                cases=build_case_index())
    except ImportError:          # code from before case-law verification
        return CitationVerifier(db, repair=True, support=True)


def report():
    from ingest.india_code import build_real_db
    from verify.verifier import Chunk
    db = build_real_db()
    ver = _verifier(db)

    print("FAIL-CLOSED PROBES (real corpus)" +
          ("" if CODE_ROOT == HERE else f"   code: {CODE_ROOT}"))
    print("=" * 78)
    rows = []
    for pid, construct, answer, keys, expect in PROBES:
        chunks = [Chunk(i, k, db.get(k).text, db.get(k).verbatim)
                  for i, k in enumerate(keys, start=1)]
        out, rep = ver.verify(answer, chunks)
        stripped = rep.sentences_stripped > 0
        hard = sorted({f.violation.value for f in rep.findings
                       if f.violation.value not in ADVISORY})
        ok = stripped == (expect == "strip")
        # A kept sentence must not come back corrupted by repair: R2's
        # enumeration bug glued a closing parenthesis to a word fragment
        # ("...of the IPC)ons 302 and 420 of the IPC)").
        if expect == "keep" and re.search(r"\)[a-z]", out):
            ok = False
        rows.append({"id": pid, "construct": construct, "expect": expect,
                     "stripped": stripped, "correct": ok, "findings": hard,
                     "output": out})
        print(f"  {'ok ' if ok else 'XX '} {pid:4} {expect:5} "
              f"{'STRIPPED' if stripped else 'kept    '} {construct[:44]:44} "
              f"{','.join(hard)}")
    faults = [r for r in rows if r["expect"] == "strip"]
    keeps = [r for r in rows if r["expect"] == "keep"]
    tp = sum(r["stripped"] for r in faults)
    tn = sum(r["correct"] for r in keeps)
    print("-" * 78)
    print(f"  faults stripped        : {tp}/{len(faults)}")
    print(f"  correct citations kept : {tn}/{len(keeps)}")
    return {"rows": rows, "faults_stripped": tp, "faults_n": len(faults),
            "kept": tn, "keep_n": len(keeps)}


def residual_behind(parser_root):
    """Defence in depth, measured: which citation-form faults the residual
    scan strips when the parser in front of it is an OLD one (parser_root),
    i.e. what the scan alone would have caught had none of this round's
    parser fixes been made. Quotation and marker faults are V4/V1's domain,
    not the scan's, and are reported separately."""
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "old_normalize", os.path.join(parser_root, "ingest", "normalize.py"))
    old = iu.module_from_spec(spec)
    spec.loader.exec_module(old)
    from verify.residual import residual_references
    out = []
    for pid, construct, answer, keys, expect in PROBES:
        if expect != "strip":
            continue
        covered = [(c.start, c.end) for c in old.extract_citations(answer)]
        covered += [(r.start, r.end) for r in old.extract_unresolved_act_refs(answer)]
        hits = residual_references(answer, covered)
        out.append({"id": pid, "construct": construct,
                    "residual_catches": bool(hits),
                    "kinds": sorted({h.kind for h in hits})})
    return out


if __name__ == "__main__":
    report()
    before = os.environ.get("BEFORE_ROOT")
    if before:
        print("\nRESIDUAL SCAN ALONE, behind the pre-revision parser")
        print("-" * 78)
        for r in residual_behind(before):
            print(f"  {r['id']:4} {'caught' if r['residual_catches'] else '  --  '}"
                  f"  {r['construct'][:46]:46} {','.join(r['kinds'])}")
