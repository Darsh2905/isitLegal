import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from verify.verifier import CitationVerifier, Chunk, Violation
from ingest.normalize import extract_citations
from eval.verifier_probes import EVASION_PROBES, CONTEXT

# Faults that must STRIP or hard-flag. V3/V5 annotate and are excluded here by
# design (CLAUDE.md invariant 2): REPEALED is included because the evasion set
# plants it as a concealed fault to be surfaced.
HARD_VIOLATIONS = (Violation.NONEXISTENT_PROVISION,
                   Violation.UNSUPPORTED_QUOTE,
                   Violation.REPEALED_PROVISION,
                   Violation.UNRESOLVED_MARKER,
                   # Added with the fail-closed revision: each strips.
                   Violation.UNKNOWN_ACT,
                   Violation.UNPLACED_REFERENCE,
                   Violation.UNKNOWN_CASE,
                   Violation.UNPARSED_REFERENCE)


def report():
    db=build_seed_db()
    ch=[Chunk(i,k,db.get(k).text,db.get(k).verbatim) for i,k in CONTEXT]
    v=CitationVerifier(db,repair=True,support=True)

    print("PARSER-EVASION RESISTANCE")
    print("="*78)
    print(f"{'probe':14}{'cites found':>12}{'flagged':>9}{'expected':>10}  {'result':>7}")
    print("-"*78)
    tp=fp=tn=fn=0
    rows=[]
    for pid,ans,bad,notes in EVASION_PROBES:
        cits=extract_citations(ans)
        out,rep=v.verify(ans,ch)
        hard=[f for f in rep.findings if f.violation in HARD_VIOLATIONS]
        flagged=len(hard)>0
        if flagged and bad: tp+=1; r="TP"
        elif flagged and not bad: fp+=1; r="FP"
        elif not flagged and not bad: tn+=1; r="TN"
        else: fn+=1; r="FN"
        rows.append({"pid": pid, "citations_found": len(cits),
                     "flagged": flagged, "expected_fault": bad, "result": r})
        print(f"{pid:14}{len(cits):>12}{'yes' if flagged else 'no':>9}"
              f"{'fault' if bad else 'clean':>10}  {r:>7}")
    print("-"*78)
    p=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}   precision={p:.2f} recall={rc:.2f}")
    print("\nWhat each probe conceals, and what was caught:")
    print("-"*78)
    caught={}
    for pid,ans,bad,notes in EVASION_PROBES:
        out,rep=v.verify(ans,ch)
        kinds=sorted({f.violation.value for f in rep.findings})
        caught[pid]=kinds
        print(f"  {pid:14} {notes[0][:44]:46} {','.join(kinds) or '-'}")

    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,
            "precision":p,"recall":rc,"rows":rows,"caught":caught}


if __name__ == "__main__":
    report()
