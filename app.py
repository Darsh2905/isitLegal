"""
Streamlit demo -- for presenting alongside the paper, not part of the research
artifact itself. Every result shown is one of three kinds, each labelled as
such: a live call into the real, already-tested deterministic code (verifier,
BM25, dense retrieval, RRF, reranking, recodification mapping); an
already-recorded real result (the generation experiment's 16 answers,
composed by this project's own model, disclosed as non-blinded); or -- since
a GROQ_API_KEY was supplied -- a genuinely live call to an independently
invoked model (Tab 6), which is real generation this project could not
demonstrate before. See generation/generator.py's module docstring for what
each client is and is not, and CLAUDE.md invariant 1 for why no model is ever
allowed inside verify/ regardless of what generates the text being checked.

Run with:  streamlit run app.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go

from ingest.metadata_db import build_seed_db
from ingest.india_code import build_real_db
from ingest.normalize import extract_citations
from retrieval.inforce import prefer_in_force
from verify.verifier import CitationVerifier, Chunk, Violation
from retrieval.bm25_baseline import BM25Retriever
from retrieval.xref import build_graph
from eval.generation_answers import UNGROUNDED_ANSWERS, GROUNDED_ANSWERS
from eval.gold_qa import GOLD
from generation.generator import (GroqClient, RetrievedChunk, generate,
                                  GeneratorNotConfigured, SYSTEM_PROMPT)

# ---------------------------------------------------------------------------
# Palette (validated categorical slot 1 + muted gray; see dataviz skill).
# Emphasis pattern: blue = the headline bar, gray = comparison/baseline.
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
GRAY = "#c3c2b7"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"

st.set_page_config(page_title="Is It Legal? — Live Demo", layout="wide",
                   page_icon="⚖️")


# ---------------------------------------------------------------------------
# Cached resources -- built once per session, reused across interactions.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading real corpus (3,275 provisions)...")
def get_real_db():
    return build_real_db()


@st.cache_resource(show_spinner="Loading toy corpus (35 provisions)...")
def get_toy_db():
    return build_seed_db()


@st.cache_resource(show_spinner="Building BM25 index over the real corpus...")
def get_bm25():
    return BM25Retriever(get_real_db())


@st.cache_resource(show_spinner="Building the gate's BM25 index...")
def get_bm25_unstemmed():
    return BM25Retriever(get_real_db(), stem=False)


@st.cache_resource
def get_verifier(repair=True, support=True):
    db = get_real_db()
    graph = build_graph(db) if support else None
    return CitationVerifier(db, repair=repair, support=support, graph=graph)


# Recalibrated separately for the real corpus by eval/run_confidence_real_eval.py
# -- NOT the toy-corpus deployment default (tau=0.305). Applying that number to
# this corpus's different score distribution reproduces a real false positive
# (see the eval script's docstring and tests/test_confidence_real.py); this is
# tau*'s own, independently-tuned value for the real 3,275-provision corpus,
# tuned on an 8/8 dev split and reported on a held-out 8/8 test split: 0/8
# unanswerable queries wrongly answered, 1/8 answerable queries wrongly
# refused (balanced accuracy 0.938). Kept in sync with the eval script by
# tests/test_confidence_real.py, not by hand.
REAL_CORPUS_TAU = 1.077


@st.cache_resource
def get_confidence_gate_real():
    from retrieval.confidence import ConfidenceGate
    # Unstemmed BM25: the gate's threshold is calibrated on that signal, and
    # stemming measurably weakens abstention (see retrieval/text.py) even
    # though it improves the retrieval shown in Tab 2.
    return ConfidenceGate(get_real_db(), get_bm25_unstemmed(), tau=REAL_CORPUS_TAU)


@st.cache_resource(show_spinner="Loading the general-purpose dense encoder (first use only, ~30-60s)...")
def get_general_dense_retriever():
    from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
    return DenseRetriever(get_real_db(), DenseEncoder(MODEL_IDS["general"]))


@st.cache_resource(show_spinner="Loading InLegalBERT (first use only, ~2-4 min — it's the slow, "
                                "underperforming arm the paper reports as a negative result)...")
def get_domain_dense_retriever():
    from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
    return DenseRetriever(get_real_db(), DenseEncoder(MODEL_IDS["domain"]))


@st.cache_resource(show_spinner="Loading the fused retriever + reranker (first use only, ~30-60s)...")
def get_fused_and_reranker():
    # Fusion uses the general arm only -- InLegalBERT (domain) is loaded
    # separately and only when Tab 2's side-by-side comparison asks for it,
    # so this path (used by Tab 2's default and Tab 3's Grounded condition)
    # doesn't pay InLegalBERT's much larger load time for a model it isn't
    # going to use.
    from retrieval.fusion import FusedRetriever
    from retrieval.rerank import Reranker
    db = get_real_db()
    general = get_general_dense_retriever()
    bm25 = get_bm25()
    fused = FusedRetriever([("bm25", bm25), ("dense", general)], candidate_k=20)
    reranker = Reranker(db)
    return fused, reranker


@st.cache_resource
def get_groq_client():
    """None if unconfigured (never raises) so the tab can render a plain
    setup message instead of crashing the whole app for anyone who clones
    this repo without a GROQ_API_KEY."""
    try:
        return GroqClient()
    except GeneratorNotConfigured:
        return None


def bar_chart(labels, values, emphasize_index, value_fmt="{:.3f}", height=320):
    """One-metric emphasis bar: blue for the headline bar, gray for the rest,
    direct value labels (secondary encoding required alongside the muted
    comparison color -- see dataviz skill). No legend: x-axis labels already
    carry category identity."""
    colors = [BLUE if i == emphasize_index else GRAY for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, width=0.5,
        text=[value_fmt.format(v) for v in values],
        textposition="outside", textfont=dict(color=INK, size=15),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=height, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif", color=SECONDARY_INK, size=14),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showline=False,
                   range=[0, max(values) * 1.25]),
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(color=SECONDARY_INK)),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚖️ Is It Legal?")
st.caption(
    "A deterministic citation verifier for Indian legal RAG — demo companion "
    "to the paper. Every result on this page is a **live call into the real, "
    "tested code** (verifier, retrieval, and — on the Live Generation tab — a "
    "real, independently invoked model), or an **already-recorded real "
    "result**, always labelled as which."
)

(tab_verify, tab_retrieve, tab_generation, tab_recode, tab_findings,
 tab_live) = st.tabs([
    "\U0001f9ea Verify Anything", "\U0001f50d Live Retrieval",
    "\U0001f4dd Real Generated Answers", "\U0001f504 Recodification Lookup",
    "\U0001f4ca The Numbers", "\U0001f680 Live Generation (Groq)",
])

# ---------------------------------------------------------------------------
# TAB 1 -- Verify Anything (the centerpiece)
# ---------------------------------------------------------------------------
with tab_verify:
    st.subheader("Paste any answer. The verifier cannot hallucinate, because it performs no inference.")
    st.markdown(
        "Write a sentence with a citation — real or made up — and press **Verify**. "
        "The check is regex, database lookups, and string matching against the "
        "real 3,275-provision corpus. No model is involved in this tab at all."
    )

    db = get_real_db()
    all_provs = db.all_provisions()
    key_options = [p.key for p in all_provs]

    def suggested_chunks(text):
        found = [c.key for c in extract_citations(text) if db.exists(c.key)][:3]
        return found or ["BNS§103", "IPC§420", "CONST§21"]

    EXAMPLES = {
        "Clean citation": "Murder is punishable under Section 103 of the Bharatiya Nyaya Sanhita [1].",
        "Fabricated section": "Murder is punishable under Section 9999 of the Bharatiya Nyaya Sanhita [1].",
        "Superseded law (auto-repaired)": "Cheating is punishable under Section 420 of the Indian Penal Code [1].",
        "Fabricated quotation": 'Article 21 provides that "no person shall ever be deprived of life without a jury trial" [1].',
        "Unresolved marker": "This is governed by the relevant provision [7].",
        "Enumerated list, one fabricated": "Sections 302 and 8888 of the IPC both apply here.",
    }
    cols = st.columns(len(EXAMPLES))
    # Both widgets below carry stable keys and are seeded once here rather than
    # via a `default=` recomputed from other state on every rerun -- a dynamic
    # default with no key is a known source of desynced widget identity across
    # reruns (surfaced by the app's own interaction test, not just a style
    # preference). An example click writes both keys explicitly instead.
    if "verify_text" not in st.session_state:
        st.session_state.verify_text = EXAMPLES["Clean citation"]
        st.session_state.retrieved_keys = suggested_chunks(st.session_state.verify_text)
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.verify_text = text
            st.session_state.retrieved_keys = [
                k for k in suggested_chunks(text) if k in key_options]

    answer = st.text_area("Answer to verify", key="verify_text", height=100)

    retrieved_keys = st.multiselect(
        "Chunks 'retrieved' for this answer (their index = the [n] marker number, in order)",
        options=key_options, key="retrieved_keys",
        help="This simulates what a retriever would have supplied to the generator.")

    if st.button("✅ Verify", type="primary"):
        by_key = {p.key: p for p in all_provs}
        chunks = [Chunk(i, k, by_key[k].text, by_key[k].verbatim)
                 for i, k in enumerate(retrieved_keys, start=1)]
        ver = get_verifier()
        verified, report = ver.verify(answer, chunks)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before**")
            st.info(answer)
        with c2:
            st.markdown("**After verification**")
            if report.has_violations():
                st.error(verified)
            else:
                st.success(verified)

        m1, m2, m3 = st.columns(3)
        m1.metric("Coverage", f"{report.coverage:.0%}")
        m2.metric("Citations checked", report.citations_total)
        m3.metric("Citations removed", report.citations_removed)

        if report.findings:
            st.markdown("**Findings**")
            for f in report.findings:
                icon = {"unresolved_marker": "\U0001f6ab", "nonexistent_provision": "\U0001f6ab",
                       "unsupported_quote": "\U0001f6ab", "repealed_provision": "⏳",
                       "invalidated_provision": "⚠️", "weak_support": "❓",
                       "support_via_reference": "\U0001f517"}.get(f.violation.value, "•")
                strips = f.violation in (Violation.UNRESOLVED_MARKER,
                                         Violation.NONEXISTENT_PROVISION,
                                         Violation.UNSUPPORTED_QUOTE)
                label = "STRIPPED" if strips else "annotated (kept)"
                st.markdown(f"{icon} **[{f.violation.value}]** — {f.detail}  \n"
                           f"&nbsp;&nbsp;&nbsp;&nbsp;→ *{label}*")
        else:
            st.markdown("No findings — every citation resolved cleanly.")

        if report.repair_log.repairs:
            st.markdown("**Auto-repair applied**")
            for r in report.repair_log.repairs:
                st.markdown(f"- `[{r.rule}]` {r.justification}")

    with st.expander("What can't be faked here?"):
        st.markdown(
            "- **A fabricated section number** is checked against a real SQLite table of "
            "3,275 real provisions — not a language model's sense of what sounds plausible.\n"
            "- **A fabricated quotation** is checked by longest-contiguous-token match "
            "against the actual retrieved text, at a 95% threshold empirically justified "
            "in the paper (character-level matching missed a meaning-inverting substitution).\n"
            "- **A superseded citation** (old IPC/CrPC/Evidence Act) is automatically "
            "rewritten to its current-law successor, sourced from an official UP Police "
            "correspondence table — try the 'Superseded law' example above.\n"
            "- **An `[n]` marker that doesn't match a retrieved chunk** is stripped, "
            "regardless of whether the cited section happens to be real."
        )

# ---------------------------------------------------------------------------
# TAB 2 -- Live Retrieval
# ---------------------------------------------------------------------------
with tab_retrieve:
    st.subheader("Real retrieval against the real 3,275-provision corpus")
    query = st.text_input("Ask a legal question",
                          value="What is the punishment for murder?")
    c1, c2 = st.columns(2)
    use_fused = c1.checkbox(
        "Dense + RRF fusion + reranking  (~30-60s on first use)")
    use_domain_compare = c2.checkbox(
        "Compare InLegalBERT vs general-purpose  (~2-4 min on first use — "
        "the paper's negative result, live)")

    if query:
        bm25 = get_bm25()
        db = get_real_db()
        by_key = {p.key: p for p in db.all_provisions()}

        def show(label, results):
            st.markdown(f"**{label}**")
            for key, score in results:
                p = by_key[key]
                st.markdown(f"`{key}` — *{p.marginal_note}* (score {score:.2f})")
                st.caption(p.text[:220] + ("..." if len(p.text) > 220 else ""))

        show("BM25 (instant, lexical)", bm25.search(query, 5))

        if use_fused:
            fused, reranker = get_fused_and_reranker()
            cands = fused.search(query, 20)
            top = reranker.rerank(query, cands, top_k=5)
            show("BM25 + Dense (general), RRF-fused, cross-encoder reranked", top)

        if use_domain_compare:
            general = get_general_dense_retriever()
            domain = get_domain_dense_retriever()
            dc1, dc2 = st.columns(2)
            with dc1:
                show("Dense — general-purpose (MiniLM)", general.search(query, 5))
            with dc2:
                show("Dense — InLegalBERT (domain-adapted)", domain.search(query, 5))
            st.caption(
                "Same query, same corpus, two embedding models. InLegalBERT is "
                "further pre-trained on Indian legal text but was never "
                "fine-tuned for retrieval/similarity — the paper's measured "
                "Recall@1 is 0.312 for InLegalBERT against 0.562 for the "
                "general model. This is usually visible above: the general "
                "arm's top hits tend to land closer to the query's actual topic."
            )

# ---------------------------------------------------------------------------
# TAB 3 -- Real Generated Answers
# ---------------------------------------------------------------------------
with tab_generation:
    st.subheader("The pre/post-verifier hallucination comparison")
    st.warning(
        "**Methodological disclosure**, stated exactly as it is in the paper: no LLM API "
        "credential is available, so these are **real, unscripted answers composed by "
        "Claude Opus 5** (the model that built this project) rather than answers from an "
        "independently invoked API call. This is not a blinded evaluation — treat it as a "
        "demonstration of correct wiring on genuine text, not a hallucination-rate estimate."
    )

    gold_lookup = {qid: (q, rel) for qid, q, rel, _ in GOLD}
    # format_func must be safe to call on its own output -- some internal
    # paths (confirmed by this app's own interaction test) re-derive a
    # widget's displayed index by re-applying format_func to the current
    # value rather than the raw option, which raises a KeyError here if the
    # function assumes it only ever sees a raw qid.
    def _format_qid(q):
        return f"{q}: {gold_lookup[q][0]}" if q in gold_lookup else q
    qid = st.selectbox("Query", options=list(gold_lookup.keys()),
                       format_func=_format_qid)
    condition = st.radio("Condition", ["Ungrounded (no retrieved context)",
                                       "Grounded (real pipeline retrieval)"],
                         horizontal=True)

    query_text, gold_keys = gold_lookup[qid]
    st.caption(f"Gold citation(s): {', '.join(gold_keys)}")

    if condition.startswith("Ungrounded"):
        answer = UNGROUNDED_ANSWERS[qid]
        chunks = []
    else:
        answer = GROUNDED_ANSWERS[qid]
        fused, reranker = get_fused_and_reranker()
        db = get_real_db()
        by_key = {p.key: p for p in db.all_provisions()}
        cands = fused.search(query_text, 20)
        top = reranker.rerank(query_text, cands, top_k=5)
        chunks = [Chunk(i, k, by_key[k].text, True) for i, (k, _s) in enumerate(top, start=1)]
        with st.expander(f"Retrieved chunks this answer was constrained to ({len(chunks)})"):
            for c in chunks:
                st.markdown(f"**[{c.index}] {c.key}**")
                st.caption(c.text[:300] + ("..." if len(c.text) > 300 else ""))

    st.markdown("**Real answer, as composed**")
    st.info(answer)

    ver = get_verifier()
    verified, report = ver.verify(answer, chunks)
    st.markdown("**After the (unmodified) verifier**")
    st.success(verified) if not report.has_violations() else st.warning(verified)
    st.caption(f"Coverage: {report.coverage:.0%}  ·  Findings: {len(report.findings)}")

# ---------------------------------------------------------------------------
# TAB 4 -- Recodification Lookup
# ---------------------------------------------------------------------------
with tab_recode:
    st.subheader("2023 recodification: old law → current law")
    st.markdown(
        "IPC→BNS, CrPC→BNSS, Evidence Act→BSA. All 11 tracked mappings are "
        "verified against a real, matched series of three UP Police correspondence tables."
    )
    db = get_real_db()
    old_provs = sorted([p.key for p in db.all_provisions() if p.repealed_by])
    picked = st.selectbox("Pick an old-law citation", options=old_provs,
                          index=old_provs.index("IPC§420") if "IPC§420" in old_provs else 0)
    if picked:
        p = db.get(picked)
        succ = db.successor(picked)
        st.markdown(f"**{picked}** — *{p.marginal_note}*")
        st.caption(p.text[:300])
        if succ:
            verified_flag = db.is_mapping_verified(picked, succ[0])
            badge = "✅ verified against the official correspondence table" if verified_flag \
                else "⚠️ not yet verified against an authoritative source"
            st.success(f"→ Current law: **{succ[0]}** — {badge}")
            sp = db.get(succ[0])
            if sp:
                st.caption(sp.text[:300])
        else:
            st.error("No successor recorded in the mapping table.")

# ---------------------------------------------------------------------------
# TAB 5 -- The Numbers
# ---------------------------------------------------------------------------
with tab_findings:
    st.subheader("Headline measured results")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retrieval de-saturation**")
        st.plotly_chart(bar_chart(
            ["Toy corpus\n(n=35)", "Real corpus\n(n=3,275)"],
            [1.000, 0.375], emphasize_index=1), use_container_width=True)
        st.caption("Plain BM25 Recall@1. The toy-corpus number is meaningless "
                  "(saturated); on the real corpus plain BM25 falls *below* the "
                  "0.49–0.60 reference range. With stemming it is 0.500; the full "
                  "deployed stack reaches 0.750.")
    with c2:
        st.markdown("**Dense retrieval: a negative result**")
        st.plotly_chart(bar_chart(
            ["BM25\n(stemmed)", "InLegalBERT\n(domain)", "General\n(MiniLM)"],
            [0.500, 0.312, 0.562], emphasize_index=2), use_container_width=True)
        st.caption("Recall@1 (dense arms scored over passage windows). A domain-adapted "
                  "legal encoder underperforms both a general-purpose retrieval-tuned "
                  "encoder and BM25 — domain pre-training without a retrieval "
                  "objective doesn't transfer.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Fusion + reranking**")
        st.plotly_chart(bar_chart(
            ["BM25", "Dense", "RRF\nFused", "+ Rerank"],
            [0.604, 0.638, 0.690, 0.821], emphasize_index=3), use_container_width=True)
        st.caption("MRR@10. Fusion beats both single arms; cross-encoder reranking over "
                  "passage windows lifts it to 0.821 (Recall@1 0.750). Reranking each "
                  "provision's first 512 tokens instead *lowers* MRR (0.676 → 0.626).")
    with c4:
        st.markdown("**Independent model (Groq): citation resolution**")
        st.plotly_chart(bar_chart(
            ["Before verifier", "After verifier"], [28 / 29, 1.000], emphasize_index=1,
            value_fmt="{:.3f}"), use_container_width=True)
        st.caption("Real answers from an independently invoked model. The one "
                  "unresolvable citation (a real section attributed to a non-existent "
                  "act) is removed; 97.2% of sentences survive verification.")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Real-corpus fault detection", "17/17",
              help="Adversarial probes on real statutory text: 17/17 faults stripped, "
                   "14/14 correct citations kept, independent audit 31/31")
    m2.metric("Repair coverage gain", "+16.7pp", help="On faulty answers, 0 invariant violations")
    m3.metric("Recodification mapping", "1,086 pairs",
              help="Full UP Police correspondence tables; 1,084 of 1,264 repealed "
                   "provisions have a verified successor")
    m4.metric("Regression suite", "278/282",
              help="The 4 failing checks assert that paper.tex states these figures; "
                   "they pass once the paper is updated")

# ---------------------------------------------------------------------------
# TAB 6 -- Live Generation (Groq): the real thing, independently invoked.
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Ask anything. A real model answers. The verifier checks it live.")
    groq_client = get_groq_client()

    if groq_client is None:
        st.error(
            "No GROQ_API_KEY configured. This tab needs one to make real "
            "generation calls — set it in `.streamlit/secrets.toml` or as an "
            "environment variable. Every other tab works without it."
        )
    else:
        st.markdown(
            f"This is a genuinely live call to **{groq_client.model}** via Groq — "
            "independently invoked, not composed by this project's own model the "
            "way Tab 3's recorded answers are. The retrieved chunks below are the "
            "*only* material the prompt allows it to use; whatever it writes is "
            "then checked by the exact same unmodified verifier as every other tab."
        )
        with st.expander("The exact system prompt it's given"):
            st.code(SYSTEM_PROMPT, language=None)

        query = st.text_input("Ask a legal question", value="",
                              placeholder="e.g. Can the police arrest me without a warrant?",
                              key="live_gen_query")

        if st.button("\U0001f680 Retrieve, Generate, Verify", type="primary"):
            if not query.strip():
                st.warning("Type a question first.")
            else:
                gate = get_confidence_gate_real()
                answerable, conf = gate.gate(query)

                if not answerable:
                    st.warning(
                        f"**Insufficient authority** — confidence "
                        f"{conf.score:.3f} is below this corpus's calibrated "
                        f"threshold ({REAL_CORPUS_TAU:.3f}). Rather than "
                        f"generate an answer from whatever the retriever "
                        f"ranked first, the gate abstains: this question "
                        f"looks like it isn't covered by these ten Acts. "
                        f"({conf.explain()})"
                    )
                else:
                    st.caption(f"Confidence gate: {conf.score:.3f} ≥ "
                              f"{REAL_CORPUS_TAU:.3f} — proceeding.")
                    db = get_real_db()
                    by_key = {p.key: p for p in db.all_provisions()}
                    fused, reranker = get_fused_and_reranker()

                    with st.spinner("Retrieving..."):
                        cands = fused.search(query, 20)
                        top = prefer_in_force(
                            reranker.rerank(query, cands, top_k=5), db)

                    gen_chunks = [RetrievedChunk(i, k, by_key[k].text)
                                 for i, (k, _s) in enumerate(top, start=1)]
                    verify_chunks = [Chunk(i, k, by_key[k].text, by_key[k].verbatim)
                                    for i, (k, _s) in enumerate(top, start=1)]

                    with st.expander(f"Retrieved chunks ({len(gen_chunks)})", expanded=False):
                        for c in gen_chunks:
                            p = by_key[c.key]
                            st.markdown(f"**[{c.index}] {c.key}** — *{p.marginal_note}*")
                            st.caption(c.text[:280] + ("..." if len(c.text) > 280 else ""))

                    with st.spinner(f"Calling {groq_client.model}..."):
                        try:
                            result = generate(query, gen_chunks, groq_client)
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            result = None

                    if result is not None:
                        st.markdown("**Raw model output**")
                        st.info(result.text)
                        if result.truncated:
                            st.warning("The model's answer was cut off at its "
                                       "length limit and may be incomplete.")

                        ver = get_verifier()
                        verified, report = ver.verify(result.text, verify_chunks)
                        st.markdown("**After the (unmodified) verifier**")
                        if report.has_violations():
                            st.warning(verified)
                        else:
                            st.success(verified)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Coverage", f"{report.coverage:.0%}")
                        m2.metric("Citations checked", report.citations_total)
                        m3.metric("Findings", len(report.findings))
                        for f in report.findings:
                            strips = f.violation in (Violation.UNRESOLVED_MARKER,
                                                     Violation.NONEXISTENT_PROVISION,
                                                     Violation.UNSUPPORTED_QUOTE)
                            st.markdown(f"- **[{f.violation.value}]** {f.detail}"
                                       f"  — *{'STRIPPED' if strips else 'annotated (kept)'}*")

        st.caption(
            "A confidence gate IS applied on this tab, recalibrated "
            "specifically for this real corpus rather than reusing the "
            "toy-corpus threshold (tau=0.305), which was validated only "
            "against 35 provisions and reproducibly answers questions this "
            "corpus should refuse. The real-corpus gate (tau="
            f"{REAL_CORPUS_TAU:.3f}) was tuned on an 8/8 dev split and "
            "reported on a held-out 8/8 test split: zero unanswerable "
            "queries wrongly answered, one answerable query wrongly "
            "refused (balanced accuracy 0.938) — see "
            "eval/run_confidence_real_eval.py."
        )
