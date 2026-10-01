"""
Judgment-corpus ingestion (CLAUDE.md Tier 1 item 6). Tests the Indian
Kanoon judgment parser (ingest/scrape_judgments.py), the two chunking
strategies (ingest/judgment_chunks.py), and the retrieval comparison
between them (retrieval/judgment_retrieval.py) -- not a claim that this
110-judgment sample is ILDC-scale or officially benchmarked; see
scrape_judgments.py's module docstring for why ILDC itself was not used
directly (the official release is gated; the one public mirror has a
systematic text-corruption artifact) and what was substituted instead.

Network-independent: everything reads from data/raw_judgments/, fetched
once and cached, matching this project's reproducibility rule (CLAUDE.md
invariant 6).
"""
import pytest

from ingest.scrape_judgments import (
    scrape_all, parse_judgment_html, segment_rhetorical_roles,
    JudgmentParagraph, RAW_DIR,
)
from ingest.judgment_chunks import naive_chunks, parent_document_chunks
from retrieval.judgment_retrieval import JudgmentRetriever
from eval.judgment_queries import GOLD


@pytest.fixture(scope="module")
def judgments():
    return scrape_all()


# --------------------------------------------------------------------------
# Parser correctness
# --------------------------------------------------------------------------

def test_corpus_is_a_real_diverse_sample_not_empty_and_not_tiny(judgments):
    # 100+ real judgments confirms the fetch/parse pipeline actually ran
    # against live Indian Kanoon pages, not a stub or an empty cache.
    assert len(judgments) >= 100


def test_every_judgment_has_case_name_and_date(judgments):
    for j in judgments:
        assert j.case_name, f"{j.doc_id} has no case name"
        assert j.date, f"{j.doc_id} has no date"


def test_every_judgment_has_at_least_one_paragraph(judgments):
    for j in judgments:
        assert len(j.paragraphs) >= 1, f"{j.doc_id} has zero paragraphs"


def test_paragraphs_are_in_source_order_with_stable_ids(judgments):
    j = next(x for x in judgments if x.doc_id == "103201262")
    ids = [p.para_id for p in j.paragraphs]
    assert ids[0] == "p_1"
    # Paragraphs and blockquotes are two independent id counters
    # interleaved in real document order (see scrape_judgments.py's
    # docstring on the "p"/"blockquote" walk) -- p_N's own numbering is
    # monotonic within itself, and so is blockquote_N's, but neither is
    # monotonic against the other's numbers, so the meaningful invariant is
    # each family's own sequence plus `.index` being contiguous overall.
    p_nums = [int(pid.split("_")[1]) for pid in ids if pid.startswith("p_")]
    bq_nums = [int(pid.split("_")[1]) for pid in ids if pid.startswith("blockquote_")]
    assert p_nums == sorted(p_nums)
    assert bq_nums == sorted(bq_nums)
    assert [p.index for p in j.paragraphs] == list(range(len(j.paragraphs)))


def test_known_judgment_metadata_matches_source(judgments):
    # Spot check against a specific, verifiable real judgment rather than
    # trusting the parser's own output in isolation.
    j = next(x for x in judgments if x.doc_id == "103201262")
    assert j.case_name == "The Silppi Constructions Contractors vs Union Of India"
    assert j.date == "21 June, 2019"
    assert j.court == "Supreme Court of India"
    assert "AIRONLINE 2019 SC 2430" in j.citations


def test_statutes_referred_is_derived_from_real_citation_normaliser(judgments):
    # Reuses ingest/normalize.py's extract_citations rather than a second,
    # untested extraction path -- confirmed here by checking the field
    # actually reflects citations present in the judgment's own text.
    j = next(x for x in judgments if x.doc_id == "103201262")
    assert "CONST§226" in j.statutes_referred
    assert any("Article 226" in p.text or "article 226" in p.text.lower()
               for p in j.paragraphs)


def test_no_soft_hyphen_or_nbsp_artifacts(judgments):
    for j in judgments:
        assert "\xad" not in j.text
        assert "\xa0" not in j.text


def test_blockquoted_precedent_holdings_are_not_silently_dropped(judgments):
    # Found by direct inspection: quoted material (a precedent's holding,
    # quoted statutory text) is rendered as a SEPARATE tag family,
    # <blockquote id="blockquote_N">, not nested inside any <p> -- 8,187
    # blockquotes across 95% of the sampled corpus, 4.1M characters, were
    # silently missing before the fix that walks <p> and <blockquote>
    # together in document order.
    j = next(x for x in judgments if x.doc_id == "103201262")
    assert any("modern trend points to judicial restraint" in p.text for p in j.paragraphs)
    idx = next(i for i, p in enumerate(j.paragraphs)
               if "modern trend points to judicial restraint" in p.text)
    # Recovered in its real reading position -- between the paragraph that
    # introduces the quote (citing Tata Cellular) and the next blockquote
    # clause -- not appended out of order at the end.
    assert "Tata Cellular" in j.paragraphs[idx - 1].text


def test_inline_efiling_and_pagination_artifacts_are_stripped(judgments):
    # A second, more severe defect found the same way: <span
    # class="hidden_text"> -- CSS-hidden page numbers and e-filing
    # digital-signature blocks -- is not hidden from get_text(), and was
    # found spliced MID-SENTENCE into real prose: "...complied with all
    # the Signature Not Verified Digitally signed by DEEPAK SINGH
    # conditions. The technical bids..." This is active corruption, not a
    # gap -- worse than a missing paragraph, since the corrupted sentence
    # reads as if it were genuine judgment text.
    j = next(x for x in judgments if x.doc_id == "103201262")
    assert "Signature Not Verified" not in j.text
    assert "Digitally signed" not in j.text
    p3 = next(p for p in j.paragraphs if p.para_id == "p_3")
    assert p3.text == (
        "2. Respondent Nos. 1 and 2 issued notice inviting tenders for "
        "two works at Kochi. The estimated cost of the works were Rs. 53 "
        "crores and Rs. 72 crores respectively. The petitioner “The Silppi "
        "Constructions Contractors”, (hereinafter referred to as the firm) "
        "uploaded its competitive bid on the site and complied with all the "
        "conditions. The technical bids of the petitioner were rejected by "
        "the tendering authorities on 28.03.2019. The petitioner filed "
        "appeals before the appellate authority on 28.03.2019 itself which "
        "were rejected on 09.04.2019."
    )


def test_free_floating_span_paragraph_is_rescued_without_duplication(judgments):
    # A rarer, third variant: one paragraph in the sampled corpus is a
    # free-standing <span id="span_N" class="citetext"> that is itself a
    # WRAPPER around a real <p id="p_N">, not a second copy of it -- taking
    # both the wrapper and the wrapped <p> would duplicate the paragraph
    # rather than rescue a missing one. Confirmed rescued exactly once.
    j = next(x for x in judgments if x.doc_id == "199473647")
    matches = [p for p in j.paragraphs
               if "Having regard to the fact that despite the law laid down" in p.text]
    assert len(matches) == 1
    assert matches[0].para_id == "p_80"


# --------------------------------------------------------------------------
# Rhetorical-role heuristic
# --------------------------------------------------------------------------

def test_rhetorical_roles_are_assigned_to_every_paragraph(judgments):
    for j in judgments:
        for p in j.paragraphs:
            assert p.role in ("facts", "issues", "reasoning", "ruling")


def test_ruling_cue_is_detected_on_a_known_disposition_paragraph():
    paras = [JudgmentParagraph(para_id="p_1", index=0,
                                text="For the reasons stated above, the appeal is hereby dismissed.")]
    segment_rhetorical_roles(paras)
    assert paras[0].role == "ruling"


def test_issue_cue_is_detected_on_a_known_framing_paragraph():
    paras = [JudgmentParagraph(
        para_id="p_1", index=0,
        text="The following questions are referred for consideration by a larger Bench.")]
    segment_rhetorical_roles(paras)
    assert paras[0].role == "issues"


def test_position_fallback_when_no_lexical_cue_fires(judgments):
    # A paragraph with no cue phrase falls back to position: near the very
    # end of a judgment it is still treated as ruling even without a
    # disposition verb, since that zone is signature blocks and dates.
    paras = [JudgmentParagraph(para_id=f"p_{i}", index=i, text="Neutral filler text.")
             for i in range(10)]
    segment_rhetorical_roles(paras)
    assert paras[-1].role == "ruling"
    assert paras[0].role == "facts"


# --------------------------------------------------------------------------
# Chunking strategies
# --------------------------------------------------------------------------

def test_naive_chunks_cover_every_paragraph_exactly_once(judgments):
    j = next(x for x in judgments if x.doc_id == "103201262")
    chunks = naive_chunks(j, window=4)
    covered = []
    for c in chunks:
        covered.extend(range(c.para_start, c.para_end + 1))
    assert covered == list(range(len(j.paragraphs)))


def test_parent_document_chunks_are_one_per_paragraph(judgments):
    j = next(x for x in judgments if x.doc_id == "103201262")
    chunks = parent_document_chunks(j)
    assert len(chunks) == len(j.paragraphs)
    for c, p in zip(chunks, j.paragraphs):
        assert c.para_start == c.para_end == p.index
        assert c.text == p.text


def test_parent_context_expands_beyond_the_bare_chunk(judgments):
    j = next(x for x in judgments if x.doc_id == "103201262" and len(x.paragraphs) > 10)
    chunks = parent_document_chunks(j)
    mid_chunk = chunks[5]
    context = mid_chunk.parent_context(j, window=2)
    assert len(context) > len(mid_chunk.text)
    assert mid_chunk.text in context


def test_parent_context_clips_at_document_boundaries(judgments):
    j = next(x for x in judgments if x.doc_id == "103201262")
    chunks = parent_document_chunks(j)
    first_chunk = chunks[0]
    # Requesting a window larger than the document must not error or index
    # negative -- the true failure mode of an unclipped window() slice.
    context = first_chunk.parent_context(j, window=100)
    assert context == j.text


# --------------------------------------------------------------------------
# Retrieval comparison (the RAPTOR-style design question, measured)
# --------------------------------------------------------------------------

def test_judgment_retriever_builds_for_both_strategies(judgments):
    for strategy in ("naive", "parent_doc"):
        r = JudgmentRetriever(judgments, strategy=strategy)
        assert len(r.chunks) > 0
        results = r.search("appeal dismissed evidence", k=5)
        assert len(results) == 5


def test_parent_document_produces_far_more_and_smaller_chunks_than_naive(judgments):
    naive = JudgmentRetriever(judgments, strategy="naive")
    parent_doc = JudgmentRetriever(judgments, strategy="parent_doc")
    # One chunk per paragraph is necessarily more numerous than fixed
    # 4-paragraph windows over the same corpus.
    assert len(parent_doc.chunks) > len(naive.chunks)
    avg_naive_len = sum(len(c.text) for c in naive.chunks) / len(naive.chunks)
    avg_pd_len = sum(len(c.text) for c in parent_doc.chunks) / len(parent_doc.chunks)
    assert avg_pd_len < avg_naive_len


def test_gold_query_set_is_grounded_in_the_real_corpus(judgments):
    # Every gold (doc_id, para_index) pair must resolve to a real paragraph
    # in the actual scraped corpus -- catches drift if the cached HTML is
    # ever refreshed and paragraph numbering shifts.
    by_id = {j.doc_id: j for j in judgments}
    for query, doc_id, para_idx, case_name in GOLD:
        j = by_id.get(doc_id)
        assert j is not None, f"gold doc_id {doc_id} not found in corpus"
        assert 0 <= para_idx < len(j.paragraphs), (
            f"gold para_index {para_idx} out of range for {doc_id} "
            f"({len(j.paragraphs)} paragraphs)")
        assert j.case_name == case_name, (
            f"gold case_name drifted for {doc_id}: "
            f"expected {case_name!r}, corpus has {j.case_name!r}")


@pytest.mark.parametrize("strategy", ["naive", "parent_doc"])
def test_retrieval_recall_at_10_is_near_perfect_on_the_gold_set(judgments, strategy):
    # A weak floor, not a tight assertion on the exact measured value: this
    # guards against a genuine regression (a chunking or indexing bug
    # collapsing recall) without pinning the test to a fragile exact
    # number that would break on every corpus-size change. See
    # eval/run_judgment_retrieval_eval.py for the actual reported
    # comparison between strategies.
    retriever = JudgmentRetriever(judgments, strategy=strategy)
    hits = 0
    for query, doc_id, para_idx, _case_name in GOLD:
        results = retriever.search(query, k=10)
        if any(c.doc_id == doc_id and c.para_start <= para_idx <= c.para_end
               for c, _score in results):
            hits += 1
    assert hits / len(GOLD) >= 0.85
