"""
Writes round2_figures.tex: the macros is_it_legal_final.tex reads for the
second live round, the support labels, verification latency and the test
count. Every value is taken from the eval scripts that measure it, so these
sections of the paper cannot drift from the code (CLAUDE.md invariant 6);
tests/test_paper_consistency.py re-derives each value and checks the file.

    python3 eval/make_paper_figures.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "round2_figures.tex")

MODEL_NAMES = {"openai/gpt-oss-20b": "gpt-oss-20b",
               "openai/gpt-oss-120b": "gpt-oss-120b",
               "qwen/qwen3.8-27b": "qwen3.8-27b"}


READABLE = {
    "unsupported_quote": "non-verbatim quotations",
    "unplaced_reference": "bare references placeable in no act",
    "unparsed_reference": "unreadable references",
    "nonexistent_provision": "nonexistent provisions",
    "unknown_act": "sections of acts outside the corpus",
    "unknown_case": "cases outside the judgment corpus",
    "unresolved_marker": "markers naming no chunk",
    "case_mismatch": "mismatched case citations",
}


def f3(x):
    return f"{x:.3f}"


def count_tests():
    # pyproject's addopts already pass -q, which suppresses the "N tests
    # collected" line here; count the collected test ids instead.
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only",
                        "-p", "no:cacheprovider", "tests/"], cwd=ROOT,
                       capture_output=True, text=True)
    return sum(1 for line in (r.stdout + r.stderr).splitlines()
               if line.startswith("tests/") and "::" in line)


def macros():
    from eval.run_live_round2_eval import report as r2
    from eval.run_support_labels_eval import report as sup
    from eval.run_latency_eval import report as lat
    live = r2()
    s = live["summary"]
    rows = live["rows"]
    n_ans = len(rows)
    queries = sorted({r["qid"] for r in rows})
    aila = [q for q in queries if q.startswith("AILA")]
    every = rows
    pre_t = sum(r["pre_total"] for r in every)
    pre_b = sum(r["pre_bad"] for r in every)
    sent = sum(r["sentences"] for r in every)
    strp = sum(r["stripped"] for r in every)
    m = {
        "RtwoAnswers": str(n_ans),
        "RtwoQueries": str(len(queries)),
        "RtwoQueriesAila": str(len(aila)),
        "RtwoPreAll": f3(1 - pre_b / pre_t),
        "RtwoPreAllCounts": f"{pre_t - pre_b}/{pre_t}",
        "RtwoCovAll": f"{100 * (1 - strp / sent):.1f}\\%",
        "RtwoSentences": str(sent),
        "RtwoResidual": str(sum(len(r["residual"]) for r in every)),
        "RtwoMissing": str(live["missing"]),
    }
    for cond in ("grounded", "ungrounded"):
        c = s[("ALL", cond)]
        k = cond.capitalize()
        m[f"Rtwo{k}Pre"] = f3(c["pre_resolution"])
        m[f"Rtwo{k}PreCounts"] = f"{c['pre_total'] - c['pre_bad']}/{c['pre_total']}"
        m[f"Rtwo{k}PreCI"] = f"{c['pre_ci'][0]:.3f}--{c['pre_ci'][1]:.3f}"
        m[f"Rtwo{k}Cov"] = f3(c["coverage"])
    g = s[("ALL", "grounded-gate-pass")]
    m["RtwoGateAnswers"] = str(g["answers"])
    m["RtwoGatePre"] = f3(g["pre_resolution"])
    m["RtwoGateCov"] = f3(g["coverage"])

    # ---- round-2 table ----
    lines = [r"\begin{table*}[t]",
             r"\caption{Second Live Round: Three Independently Invoked Models, "
             f"{len(aila)} AILA Fact Situations $+$ {len(queries) - len(aila)} "
             r"Gold Questions. Resolution is "
             r"re-derived by an audit independent of the verifier; 95\% Wilson "
             r"intervals.}",
             r"\label{tab:live2}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{@{}llrrcccr@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{Condition} & \textbf{Answers} & "
             r"\textbf{Cut off} & \textbf{Resolution, pre} & "
             r"\textbf{Resolution, post} & \textbf{Coverage} & "
             r"\textbf{Residual} \\", r"\midrule"]
    for cond in ("grounded", "ungrounded"):
        for model in MODEL_NAMES:
            c = s.get((model, cond))
            if not c:
                continue
            lines.append(
                f"{MODEL_NAMES[model]} & {cond} & {c['answers']} & {c['truncated']} & "
                f"{f3(c['pre_resolution'])} ({c['pre_total']-c['pre_bad']}/{c['pre_total']}; "
                f"{c['pre_ci'][0]:.2f}--{c['pre_ci'][1]:.2f}) & "
                f"{f3(c['post_resolution'])} ({c['post_total']-c['post_bad']}/{c['post_total']}) & "
                f"{f3(c['coverage'])} & {c['residual_hits']} \\\\")
        c = s[("ALL", cond)]
        lines.append(
            f"\\textbf{{All}} & {cond} & {c['answers']} & {c['truncated']} & "
            f"\\textbf{{{f3(c['pre_resolution'])}}} ({c['pre_total']-c['pre_bad']}/{c['pre_total']}; "
            f"{c['pre_ci'][0]:.2f}--{c['pre_ci'][1]:.2f}) & "
            f"\\textbf{{{f3(c['post_resolution'])}}} ({c['post_total']-c['post_bad']}/{c['post_total']}) & "
            f"\\textbf{{{f3(c['coverage'])}}} & {c['residual_hits']} \\\\")
        if cond == "grounded":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    m["RtwoTable"] = "\n".join(lines)
    for cond in ("grounded", "ungrounded"):
        c = s[("ALL", cond)]
        k = cond.capitalize()
        m[f"Rtwo{k}StrippedBy"] = ", ".join(
            f"{v:,} {READABLE.get(kk, kk.replace('_', ' '))}" for kk, v in
            sorted(c["stripped_by"].items(), key=lambda t: -t[1]))

    # ---- support ----
    sp = sup()
    bc = sp["by_condition"]
    m["SupN"] = str(sp["n_sample"])
    sl = [r"\begin{table}[t]",
          r"\caption{Support of Surviving Cited Sentences (author-labelled "
          r"random sample; 95\% Wilson intervals)}",
          r"\label{tab:support-labels}", r"\centering", r"\footnotesize",
          r"\begin{tabular}{@{}lrrrrc@{}}", r"\toprule",
          r"\textbf{Condition} & \textbf{n} & \textbf{Sup.} & \textbf{Part.} & "
          r"\textbf{Unsup.} & \textbf{Fully supported} \\", r"\midrule"]
    for cond in ("grounded", "ungrounded", "all"):
        c = bc.get(cond)
        if not c:
            continue
        lo, hi = c["support_precision_ci"]
        sl.append(f"{cond.capitalize()} & {c['n']} & {c['supported']} & "
                  f"{c['partial']} & {c['unsupported']} & "
                  f"{f3(c['support_precision'])} ({lo:.2f}--{hi:.2f}) \\\\")
    sl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    m["SupTable"] = "\n".join(sl)
    for cond in ("grounded", "ungrounded", "all"):
        c = bc.get(cond)
        if not c:
            continue
        k = cond.capitalize()
        m[f"Sup{k}Prec"] = f3(c["support_precision"])
        m[f"Sup{k}CI"] = (f"{c['support_precision_ci'][0]:.2f}--"
                          f"{c['support_precision_ci'][1]:.2f}")
        m[f"Sup{k}OrPartial"] = f3(c["support_or_partial"])
        m[f"Sup{k}Unsup"] = str(c["unsupported"])
        m[f"Sup{k}N"] = str(c["n"])
    a = bc["all"]
    m["SupVfiveCaught"] = f"{a['v5_true_flags']} of {a['unsupported']}"
    m["SupVfiveFlagged"] = str(a["v5_flagged"])
    m["SupVfiveTrue"] = str(a["v5_true_flags"])
    m["SupVfiveFlagPrec"] = f3(a["v5_true_flags"] / a["v5_flagged"]) if a["v5_flagged"] else "n/a"

    # ---- latency, tests ----
    lt = lat()
    m["LatencyN"] = str(lt["n"])
    m["LatencyMedian"] = f"{lt['median_ms']:.1f}"
    m["LatencyPninetyfive"] = f"{lt['p95_ms']:.1f}"
    m["LatencyMax"] = f"{lt['max_ms']:.1f}"
    m["TestCount"] = str(count_tests())
    return m


def main():
    m = macros()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% Generated by eval/make_paper_figures.py -- do not edit by hand.\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {OUT} ({len(m)} macros)")
    return m


if __name__ == "__main__":
    main()
