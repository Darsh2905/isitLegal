"""
End-to-end query pipeline: Stages 1-7 of the architecture, wired together.

STATUS. This is the first point in the project where every stage the paper's
architecture diagram promises actually exists in code -- BM25 + dense
retrieval (Stage 2), RRF fusion, the confidence gate (Stage 4), generation
(Stage 5), citation verification (Stage 6), response assembly (Stage 7).
Whether it has ever been RUN with a real model is a separate question: see
generation/generator.py. `answer_query` below accepts any LLMClient, so it can
be exercised completely with a scripted fake (tests/test_pipeline.py) or, if
ANTHROPIC_API_KEY is set, with a real one -- the code path does not change
between the two, which is the point of the dependency injection.

Stage 1 (query understanding / rewriting) is not implemented. Stage 3
(cross-encoder reranking) is optional: pass `reranker=` to enable it. It is
the best-measured configuration (eval/run_rerank_eval.py) but costs a
cross-encoder pass per query, so the default leaves it off.
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest.metadata_db import MetadataDB
from retrieval.confidence import ConfidenceGate, ABSTENTION_MESSAGE
from retrieval.fusion import FusedRetriever
from retrieval.inforce import prefer_in_force
from verify.verifier import CitationVerifier, Chunk, VerificationReport
from generation.generator import (RetrievedChunk, PromptBuilder, generate,
                                  LLMClient, GeneratedAnswer)


@dataclass
class PipelineResult:
    query: str
    answered: bool
    answer: str
    confidence: Optional[float] = None
    retrieved_keys: Optional[List[str]] = None
    raw_generation: Optional[str] = None
    verification: Optional[VerificationReport] = None
    truncated: bool = False


class Pipeline:
    """
    Stage 1 [minimal] -> Stage 2 (BM25+dense, RRF-fused) -> Stage 4 (confidence
    gate) -> Stage 5 (generation) -> Stage 6 (verification) -> Stage 7
    (response assembly, minimal: this returns the verified text plus its
    report rather than rendering source cards).
    """

    def __init__(self, db: MetadataDB, retriever: FusedRetriever,
                gate: ConfidenceGate, client: LLMClient,
                verifier: Optional[CitationVerifier] = None, top_k: int = 5,
                reranker=None, candidate_k: int = 20):
        self.db = db
        self.retriever = retriever
        self.gate = gate
        self.client = client
        if verifier is None:
            # Case citations resolve against the judgment corpus when its
            # cache is present (ingest/scrape_judgments.py); with an empty
            # index every case citation is stripped, the closed-world default.
            from ingest.case_citations import build_case_index
            verifier = CitationVerifier(db, repair=True, support=True,
                                        cases=build_case_index())
        self.verifier = verifier
        self.top_k = top_k
        # Optional Stage 3. With passage-window scoring (retrieval/rerank.py)
        # it is the best-measured configuration on the real corpus (MRR@10
        # 0.821 vs 0.683 fused alone); optional because it adds a
        # cross-encoder pass per query.
        self.reranker = reranker
        self.candidate_k = candidate_k

    def answer_query(self, query: str) -> PipelineResult:
        # Stage 4: abstain before spending a generation call on a query the
        # corpus cannot support.
        answerable, conf = self.gate.gate(query)
        if not answerable:
            return PipelineResult(query=query, answered=False,
                                  answer=ABSTENTION_MESSAGE,
                                  confidence=conf.score, retrieved_keys=[])

        # Stage 2: fused retrieval, then in-force preference (a repealed
        # provision's recorded successor is moved ahead of it; nothing is
        # added or removed -- retrieval/inforce.py).
        if self.reranker is not None:
            cands = self.retriever.search(query, self.candidate_k)
            ranked = self.reranker.rerank(query, cands, top_k=self.top_k)
        else:
            ranked = self.retriever.search(query, self.top_k)
        results = prefer_in_force(ranked, self.db)
        chunks_for_gen = []
        chunks_for_verify = []
        for i, (key, _score) in enumerate(results, start=1):
            # Look up only the top-k provisions -- this previously loaded the
            # entire corpus on every query.
            prov = self.db.get(key)
            if prov is None:
                continue
            chunks_for_gen.append(RetrievedChunk(index=i, key=key, text=prov.text))
            chunks_for_verify.append(Chunk(index=i, key=key, text=prov.text,
                                           verbatim=prov.verbatim))

        if not chunks_for_gen:
            return PipelineResult(query=query, answered=False,
                                  answer=ABSTENTION_MESSAGE,
                                  confidence=conf.score, retrieved_keys=[])

        # Stage 5: generation.
        generated: GeneratedAnswer = generate(query, chunks_for_gen, self.client)
        if generated.truncated:
            # Surfaced, not silently verified as if complete: a completion cut
            # off at the token limit can end mid-claim or mid-quotation.
            truncated_note = ("\n\nNote: the generated answer was cut off at the "
                              "model's length limit and may be incomplete.")
        else:
            truncated_note = ""

        # Stage 6: verification. Stage 7: response assembly (minimal).
        verified_text, report = self.verifier.verify(generated.text, chunks_for_verify)

        return PipelineResult(
            query=query, answered=True, answer=verified_text + truncated_note,
            truncated=generated.truncated,
            confidence=conf.score,
            retrieved_keys=[c.key for c in chunks_for_verify],
            raw_generation=generated.text, verification=report)
