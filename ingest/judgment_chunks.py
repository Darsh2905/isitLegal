"""
Two chunking strategies over the judgment corpus (ingest/scrape_judgments.py),
built so the RAPTOR-style design decision named in the architecture --
"the paragraph is the retrieval unit, but the surrounding context is
supplied to the generator" -- can actually be measured instead of only
argued for by citation (CLAUDE.md Tier 1 item 6).

NAIVE chunking indexes fixed-size windows of consecutive paragraphs (no
awareness of paragraph boundaries as meaningful units, no separate notion of
"the chunk that matched" vs "the context to hand the generator") -- the
flat, RAPTOR-agnostic alternative.

PARENT-DOCUMENT chunking indexes each paragraph individually (small, precise
retrieval units, so a query about one specific holding doesn't have to
compete lexically against an entire judgment's unrelated surrounding prose)
but each chunk carries a pointer back to its parent judgment, so a consumer
can expand a single matched paragraph into its full surrounding context
before handing it to a generator. This is the strategy the paper's
architecture section commits to; NAIVE exists so that commitment is a
measured comparison rather than an assumption.
"""

from dataclasses import dataclass
from typing import List

from ingest.scrape_judgments import Judgment, JudgmentParagraph


@dataclass
class JudgmentChunk:
    chunk_id: str
    doc_id: str
    case_name: str
    text: str              # what gets indexed and scored
    para_start: int         # first paragraph index covered (inclusive)
    para_end: int           # last paragraph index covered (inclusive)

    def parent_context(self, judgment: "Judgment", window: int = 2) -> str:
        """The chunk's text expanded with `window` paragraphs of surrounding
        context on each side, clipped to the judgment's own bounds -- what a
        generator would actually be handed under parent-document retrieval,
        not just the bare matched chunk."""
        lo = max(0, self.para_start - window)
        hi = min(len(judgment.paragraphs) - 1, self.para_end + window)
        return " ".join(p.text for p in judgment.paragraphs[lo:hi + 1])


def naive_chunks(judgment: Judgment, window: int = 4) -> List[JudgmentChunk]:
    """Fixed-size windows of `window` consecutive paragraphs, non-overlapping.
    The last window in a judgment may be shorter than `window` paragraphs."""
    chunks = []
    paras = judgment.paragraphs
    for start in range(0, len(paras), window):
        group = paras[start:start + window]
        if not group:
            continue
        text = " ".join(p.text for p in group)
        chunks.append(JudgmentChunk(
            chunk_id=f"{judgment.doc_id}#naive:{start}",
            doc_id=judgment.doc_id, case_name=judgment.case_name, text=text,
            para_start=group[0].index, para_end=group[-1].index))
    return chunks


def parent_document_chunks(judgment: Judgment) -> List[JudgmentChunk]:
    """One chunk per paragraph -- the small, precise retrieval unit that
    parent_context() above later expands."""
    chunks = []
    for p in judgment.paragraphs:
        chunks.append(JudgmentChunk(
            chunk_id=f"{judgment.doc_id}#{p.para_id}",
            doc_id=judgment.doc_id, case_name=judgment.case_name, text=p.text,
            para_start=p.index, para_end=p.index))
    return chunks
