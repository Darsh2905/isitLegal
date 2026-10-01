"""
Constrained generation (Stage 5) -- the piece the verifier exists to check.

STATUS. `AnthropicClient` remains uncallable: no ANTHROPIC_API_KEY is
configured in this environment. `GroqClient` IS live-callable -- a
GROQ_API_KEY was supplied and verified against a real completion -- which
makes an INDEPENDENTLY invoked model available, rather than only the
scripted fake in tests/test_generator.py or the answers in
eval/generation_answers.py that this project's own model composed. That
capability is wired into app.py's "Live Generation" tab (Tab 6) for
interactive demonstration, AND into eval/run_live_generation_eval.py for a
paper-facing measurement: the two are not the same thing. A single
interactive Tab 6 query can never back a paper number (CLAUDE.md invariant
6 -- every claim must be reproducible without a paid credential), so
eval/run_live_generation_eval.py instead freezes the one genuinely
non-reproducible part (the raw Groq completion text, captured once in
eval/live_generation_answers.py) and re-derives everything else --
confidence-gate decision, retrieval, verification -- fresh from the live
corpus and current code on every run, the same split every other eval/
script draws between what must be frozen and what must not. See
eval/live_generation_answers.py's docstring for exactly what that
experiment establishes and does not: real, independent generation, still
over a project-authored query set.

WHY THE INTERFACE WAS BUILT PROVIDER-AGNOSTIC. `LLMClient` is a narrow
Protocol any provider's SDK can satisfy, `PromptBuilder` constructs the
constrained prompt from retrieved chunks, and `generate()` turns a raw model
response back into the (answer, Chunk-list) shape
verify.verifier.CitationVerifier.verify expects. tests/test_generator.py
exercises all of this against a scripted FAKE client -- a stand-in that
returns pre-written strings, not a model -- which proves the WIRING is
correct independent of which provider ends up plugged in. That the two real
clients below (Anthropic, Groq) share one interface and neither needed
pipeline.py to change is the payoff of that choice.

THE PROMPT CONTRACT. Forbids outside knowledge, requires an inline [n] marker
naming the retrieved chunk for every claim, and requires verbatim quotation in
quotation marks for any claim purporting to quote source text -- exactly what
verify.verifier.CitationVerifier checks for. The contract and the checker are
deliberately written to match: the generator is asked to do what the verifier
can confirm, not asked to do more and hope.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Protocol


class GeneratorNotConfigured(RuntimeError):
    """
    Raised by `AnthropicClient` (or any real client) when no API credential is
    present, rather than silently returning empty or fabricated text. A
    generator that fails loudly when unconfigured is safer than one that
    fails by producing plausible-looking nothing.
    """


class LLMClient(Protocol):
    """Narrow interface any provider's SDK can satisfy. One method: given a
    system prompt and a user prompt, return the raw completion text."""

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class RetrievedChunk:
    index: int
    key: str
    text: str


SYSTEM_PROMPT = """You are a legal-citation assistant restricted to the supplied source material.

Rules, followed exactly:
1. Use ONLY the numbered source chunks provided. Do not draw on any other knowledge of Indian law, real or invented.
2. Every factual claim must end with an inline marker [n] naming the chunk it comes from.
3. If you quote source text verbatim, enclose it in double quotation marks and reproduce it exactly -- no paraphrasing inside quotation marks.
4. If the supplied chunks do not answer the question, say so plainly rather than guessing or filling the gap with general knowledge.
5. Do not cite a chunk number that was not supplied."""


class PromptBuilder:
    @staticmethod
    def build(query: str, chunks: List[RetrievedChunk]) -> str:
        lines = [f"Question: {query}", "", "Source chunks:"]
        for c in chunks:
            lines.append(f"[{c.index}] {c.key}: {c.text}")
        lines.append("")
        lines.append("Answer, following the rules exactly.")
        return "\n".join(lines)


@dataclass
class GeneratedAnswer:
    text: str
    raw_response: str
    # True when the provider reports the completion stopped at max_tokens.
    # Clients expose this as `last_finish_reason`; clients that do not (the
    # scripted test fakes) are treated as complete.
    truncated: bool = False


def generate(query: str, chunks: List[RetrievedChunk], client: LLMClient) -> GeneratedAnswer:
    """
    The only function that matters for pipeline wiring: given a query, the
    chunks retrieval supplied, and ANY object satisfying LLMClient, produce a
    GeneratedAnswer. `client` is dependency-injected specifically so tests can
    pass a scripted fake instead of a real model -- see tests/test_generator.py.
    """
    prompt = PromptBuilder.build(query, chunks)
    raw = client.complete(SYSTEM_PROMPT, prompt)
    truncated = getattr(client, "last_finish_reason", None) in ("length", "max_tokens")
    return GeneratedAnswer(text=raw.strip(), raw_response=raw, truncated=truncated)


class AnthropicClient:
    """
    Real client, built to the same LLMClient interface. Constructing it
    without ANTHROPIC_API_KEY set raises GeneratorNotConfigured immediately --
    it does not defer the failure to the first `.complete()` call, and it does
    not fall back to fabricating a response.
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise GeneratorNotConfigured(
                "ANTHROPIC_API_KEY is not set. This project does not fabricate "
                "generated output when unconfigured -- set the key or supply "
                "a different LLMClient.")
        try:
            import anthropic
        except ImportError as e:
            raise GeneratorNotConfigured(
                "the `anthropic` package is not installed") from e
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user}])
        self.last_finish_reason = getattr(resp, "stop_reason", None)
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


class GroqClient:
    """
    Real client for models served over Groq's API, built to the same
    LLMClient interface as AnthropicClient. Constructing it without
    GROQ_API_KEY set (as an argument, an environment variable, or --
    Streamlit-specific -- st.secrets) raises GeneratorNotConfigured
    immediately, the same fail-loudly contract AnthropicClient uses.

    DEFAULT MODEL. `openai/gpt-oss-20b`, chosen empirically, not assumed. An
    earlier version of this project recommended `llama-3.3-70b-versatile`,
    following outdated training-era knowledge of Groq's lineup; that model
    is not in the catalog this key can see, and model catalogs on inference
    platforms change often enough that an identifier should be verified
    against `client.models.list()` before being trusted at all. The next
    candidate, the larger `openai/gpt-oss-120b`, DID work, but on the same
    prompt this project's system prompt uses (Section 2: "every factual
    claim must end with an inline marker [n]") it substituted its own
    native citation-annotation style (`【†...】`, an
    OpenAI-family file-citation format) instead of the instructed `[n]`
    markers on a live test call -- output that would silently carry ZERO
    citations as far as this project's regex-based marker extraction is
    concerned, passing every check not because it verified correctly but
    because the check never saw a citation to verify. The smaller
    `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` both complied with the
    instructed format exactly, including verbatim quotation, on the same
    live test; `qwen/qwen3.6-27b` is a reasoning model that emits a
    `<think>` block before its answer and needs a larger token budget than
    this project allocates, cut off mid-thought at this call's max_tokens.
    Bigger is not better here, and the failure mode a bigger model produced
    is exactly the parser-evasion class this project's own methodology
    (CLAUDE.md Section 6) already has a name for -- caught only because
    output was actually inspected rather than assumed compliant.
    """

    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        if api_key is None:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
        if not api_key:
            raise GeneratorNotConfigured(
                "GROQ_API_KEY is not set (checked: argument, environment, "
                "st.secrets). This project does not fabricate generated "
                "output when unconfigured -- set the key or supply a "
                "different LLMClient.")
        try:
            import groq
        except ImportError as e:
            raise GeneratorNotConfigured(
                "the `groq` package is not installed") from e
        self._client = groq.Groq(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=1024,
            messages=[{"role": "system", "content": system},
                     {"role": "user", "content": user}])
        self.last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
        return resp.choices[0].message.content or ""
