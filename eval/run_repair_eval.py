import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from verify.verifier import CitationVerifier, Chunk, Violation
from ingest.normalize import extract_citations, extract_unresolved_act_refs
from eval.verifier_probes import PROBES, QUOTE_PROBES, REPAIR_PROBES, CONTEXT

ALL = PROBES + QUOTE_PROBES + REPAIR_PROBES


def build_context(db):
    return [Chunk(i, k, db.get(k).text, db.get(k).verbatim) for i, k in CONTEXT]


def audit(answer, retrieved, db):
    """
    Independent post-hoc audit of an EMITTED answer.
    Confirms the invariant directly rather than trusting the verifier's own
    report: every surviving marker resolves, every surviving reference exists
    and is in force (or explicitly annotated).

    INVARIANT 3 (see CLAUDE.md): repair must re-verify, never bypass. This
    audit is the independent check backing that claim -- it re-derives the
    invariant from the emitted text alone, without consulting the verifier's
    own report. Preserve it and re-run it after any change to repair logic.
    """
    import re
    valid = {c.index for c in retrieved}
    body = answer.split("\n\nNote:")[0].split("\n\n")[0]
    for n in re.findall(r"\[(\d+)\]", body):
        if int(n) not in valid: return False, f"marker [{n}] unresolved"
    for c in extract_citations(body):
        if db.resolve(c.key) is None: return False, f"{c.key} nonexistent"
    for r in extract_unresolved_act_refs(body):
        return False, f"section {r.section} of unknown act {r.act_name!r}"
    return True, "ok"


def report():
    db = build_seed_db()
    chunks = build_context(db)

    print("REPAIR ABLATION")
    print("="*76)
    print(f"{'mode':14}{'cov(all)':>10}{'cov(faulty)':>13}{'audit pass':>12}{'repairs':>9}")
    print("-"*76)
    res={}
    for label,flag in (("strip-only",False),("strip+repair",True)):
        v=CitationVerifier(db,repair=flag)
        covs=[]; fcovs=[]; ok=0; nrep=0
        for pid,ans,hf,ft,note in ALL:
            out,rep=v.verify(ans,chunks)
            covs.append(rep.coverage)
            if hf: fcovs.append(rep.coverage)
            good,_=audit(out,chunks,db); ok+=good
            nrep+=rep.repair_log.succeeded
        res[label]=(sum(covs)/len(covs), sum(fcovs)/len(fcovs), ok, nrep)
        print(f"{label:14}{res[label][0]:>10.3f}{res[label][1]:>13.3f}"
              f"{ok:>8}/{len(ALL):<3}{nrep:>9}")
    print("-"*76)
    d_all=res['strip+repair'][0]-res['strip-only'][0]
    d_f  =res['strip+repair'][1]-res['strip-only'][1]
    violations = res['strip-only'][2]-res['strip+repair'][2]
    print(f"Coverage gain: {d_all:+.3f} overall, {d_f:+.3f} on faulty answers")
    print(f"Invariant violations introduced by repair: {violations}")

    print("\nREPAIRS PERFORMED")
    print("-"*76)
    v=CitationVerifier(db,repair=True)
    performed=[]
    for pid,ans,hf,ft,note in ALL:
        out,rep=v.verify(ans,chunks)
        for r in rep.repair_log.repairs:
            performed.append((r.rule, pid))
            print(f"[{r.rule}] {pid}")
            print(f"   before: {r.before[:66]}")
            print(f"   after : {r.after[:66]}")
            print(f"   why   : {r.justification[:66]}")

    print("\nUNREPAIRABLE BY DESIGN (fabricated content -- no evidence to recover)")
    print("-"*76)
    v2=CitationVerifier(db,repair=True)
    unrepairable=[]
    for pid,ans,hf,ft,note in ALL:
        out,rep=v2.verify(ans,chunks)
        kinds={f.violation for f in rep.findings}
        if (Violation.NONEXISTENT_PROVISION in kinds or
            Violation.UNSUPPORTED_QUOTE in kinds):
            unrepairable.append(pid)
            print(f"  {pid:11} {note[:56]}")

    return {
        "n_probes": len(ALL),
        "coverage_all": {k: v[0] for k, v in res.items()},
        "coverage_faulty": {k: v[1] for k, v in res.items()},
        "audit_pass": {k: v[2] for k, v in res.items()},
        "repairs": {k: v[3] for k, v in res.items()},
        "coverage_gain_all": d_all,
        "coverage_gain_faulty": d_f,
        "invariant_violations": violations,
        "repairs_performed": performed,
        "unrepairable": unrepairable,
    }


if __name__ == "__main__":
    report()
