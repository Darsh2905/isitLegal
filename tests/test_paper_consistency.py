"""
Paper/code consistency (CLAUDE.md invariant 6) for the final paper,
is_it_legal_final.tex.

Every quantitative claim in the paper must be reproducible by a script in
eval/. Each assertion takes a figure from the harness, formats it as the
paper writes it, and requires it to appear in the source; a measurement that
changes fails here until the paper is updated in the same edit. Figures from
the second live round, the support labels, latency and the test count are
not typed into the paper at all: eval/make_paper_figures.py writes them to
round2_figures.tex, which the paper \\input{}s, and the last tests check that
file against a fresh measurement.

(paper.tex, the pre-revision draft, is no longer checked: it is superseded.)
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "is_it_legal_final.tex")
FIGURES = os.path.join(ROOT, "round2_figures.tex")


@pytest.fixture(scope="module")
def paper():
    if not os.path.exists(PAPER):
        pytest.skip("is_it_legal_final.tex not present in this checkout")
    with open(PAPER, encoding="utf-8") as fh:
        # Whitespace-normalised: a phrase may wrap across a source line.
        return re.sub(r"\s+", " ", fh.read())


@pytest.fixture(scope="module")
def figures():
    if not os.path.exists(FIGURES):
        pytest.skip("round2_figures.tex not generated")
    out = {}
    with open(FIGURES, encoding="utf-8") as fh:
        text = fh.read()
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", text):
        start, depth, i = m.end(), 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out[m.group(1)] = text[start:i - 1]
    return out


def _require(paper, needle, what):
    assert needle in paper, (
        f"the paper does not contain {what} as measured ({needle!r}). The "
        f"measurement changed and the paper was not updated in the same edit "
        f"(CLAUDE.md invariant 6).")


def _tex(n):
    return f"{n:,}".replace(",", "{,}") if False else f"{n:,}"


# ---- corpora ------------------------------------------------------------------

def test_real_corpus_size(paper, real_corpus_report):
    _require(paper, f"{real_corpus_report['corpus_size']:,} provisions",
             "the real corpus size")
    _require(paper, f"{real_corpus_report['corpus_size']:,}-provision",
             "the real corpus size (adjectival)")


def test_real_graph_and_mapping(paper, real_corpus_report):
    r = real_corpus_report
    _require(paper, f"{r['graph_edges']:,} edges", "the real citation-graph size")
    _require(paper, f"marks {r['non_self_contained_real']} provisions",
             "non-self-contained provisions")
    _require(paper, f"{r['mapping_full_pairs']:,} section-level pairs", "mapping pairs")
    _require(paper, f"{r['repealed_with_successor']:,} of the {r['repealed_total']:,} "
                    f"repealed provisions", "mapping coverage")


def test_development_corpus_size(paper, retrieval):
    _require(paper, f"{retrieval['corpus_size']}-provision development corpus",
             "the development corpus size")


# ---- retrieval -----------------------------------------------------------------

def test_real_retrieval_table(paper, dense, fusion):
    r = fusion["results"]
    _require(paper, f"\\textbf{{{r['bm25+rrf+dense'][1]:.3f}}} & \\textbf{{{r['bm25+rrf+dense'][10]:.3f}}}"
             f" & \\textbf{{{r['bm25+rrf+dense']['mrr@10']:.3f}}}", "the fused row")
    _require(paper, f"{r['bm25'][1]:.3f} & {r['bm25'][10]:.3f} & {r['bm25']['mrr@10']:.3f}",
             "the stemmed BM25 row")


def test_rerank_table(paper, rerank):
    r = rerank["results"]
    _require(paper, f"{r['bm25+rrf+dense']['mrr@10']:.3f} & {r['bm25+rrf+dense']['ndcg@10']:.3f}",
             "the fused row of the rerank table")
    _require(paper, f"\\textbf{{{r['+ rerank']['mrr@10']:.3f}}}", "reranked MRR@10")


def test_aila_table(paper, aila):
    for name, row in aila["systems"].items():
        _require(paper, f"{row['map']:.4f} & {row['bpref']:.4f} & "
                        f"{row['recip_rank']:.4f} & {row['p10']:.4f}",
                 f"the AILA row for {name}")
    assert aila["n_test"] == 40


def test_inforce_diagnostic(paper, inforce):
    real = inforce["real"]
    _require(paper, f"in only {real['BM25 stemmed']} of "
                    f"{real['n_pairs']} pairs", "BM25 in-force count")


# ---- verifier ------------------------------------------------------------------

def test_real_probe_table(paper, verifier_real):
    _require(paper, f"\\textbf{{{verifier_real['tp']}/{verifier_real['tp'] + verifier_real['fn']}}}",
             "round-1 faults stripped")
    _require(paper, f"\\textbf{{{verifier_real['tn']}/{verifier_real['tn'] + verifier_real['fp']}}}",
             "round-1 correct kept")


def test_failclosed_table(paper, failclosed):
    _require(paper, f"\\textbf{{{failclosed['faults_stripped']}/{failclosed['faults_n']}}}",
             "fail-closed faults stripped")
    _require(paper, f"\\textbf{{{failclosed['kept']}/{failclosed['keep_n']}}}",
             "fail-closed correct kept")
    total_faults = failclosed["faults_n"] + 17
    total_keep = failclosed["keep_n"] + 14
    _require(paper, f"all {total_faults} planted faults", "abstract fault total")
    _require(paper, f"any of {total_keep} correct citations", "abstract keep total")


def test_quote_length_sweep_table(paper, quote):
    ls = quote["length_sweep"]
    for n in (10, 19, 20, 30, 40, 60):
        r = ls[(0.95, n)]
        _require(paper, f"{r['substitution_accepted']}/{r['substitution_n']}",
                 f"theta=0.95 acceptance at n={n}")
        r = ls[(1.0, n)]
        assert r["substitution_accepted"] == 0
        _require(paper, f"\\textbf{{0/{r['substitution_n']}}}", f"exact rule at n={n}")


def test_live_round_one(paper, live_generation):
    lg = live_generation
    _require(paper, f"{lg['pre_accuracy']:.3f} ({lg['pre_total'] - lg['pre_bad']}/{lg['pre_total']})",
             "round-1 pre-verifier resolution")
    _require(paper, f"1.000 ({lg['post_total']}/{lg['post_total']})",
             "round-1 post-verifier resolution")
    _require(paper, f"{100 * lg['mean_coverage']:.1f}\\% of sentences kept",
             "round-1 coverage")


def test_gate_figures(paper, confidence_real, confidence_external):
    _require(paper, f"\\tau_r = {confidence_real['tau']:.3f}", "the real-corpus tau")
    _require(paper, f"({confidence_external['correct_abstention']:.3f})",
             "external abstention")


def test_provenance(paper, provenance):
    _require(paper, f"{provenance['matching']:,} of the {provenance['tested']:,} testable sections",
             "the Gazette agreement count")
    for act, v in provenance["acts"].items():
        _require(paper, f"{act} {v['at_or_above']}/{v['tested']}", f"{act} agreement")


def test_mapping_audit(paper, mapping_audit):
    m = mapping_audit
    _require(paper, f"Of the {m['both']} IPC sections both tables map", "mapping overlap")
    _require(paper, f"for {m['exact']} ({m['exact'] / m['both']:.3f})", "mapping agreement")
    assert m["disagree"] == 0


# ---- generated figures -------------------------------------------------------

def test_paper_reads_generated_figures(paper):
    _require(paper, "\\input{round2_figures}", "the generated figures file")


def test_generated_figures_match_a_fresh_measurement(figures, live_round2, support_labels):
    s = live_round2["summary"]
    rows = live_round2["rows"]
    pre_t = sum(r["pre_total"] for r in rows)
    pre_b = sum(r["pre_bad"] for r in rows)
    assert figures["RtwoAnswers"] == str(len(rows))
    assert figures["RtwoPreAll"] == f"{1 - pre_b / pre_t:.3f}"
    for cond in ("grounded", "ungrounded"):
        assert figures[f"Rtwo{cond.capitalize()}Pre"] == f"{s[('ALL', cond)]['pre_resolution']:.3f}"
        assert s[("ALL", cond)]["post_bad"] == 0, "post-verifier resolution below 1.0"
    sup = support_labels["by_condition"]
    assert figures["SupN"] == str(support_labels["n_sample"])
    assert figures["SupGroundedPrec"] == f"{sup['grounded']['support_precision']:.3f}"
    assert support_labels["n_missing"] == 0, "sampled sentences left unlabelled"


def test_generated_latency_is_current(figures, latency):
    # Wall-clock: compared within a factor of two, not exactly.
    assert 0.5 <= float(figures["LatencyMedian"]) / latency["median_ms"] <= 2.0
