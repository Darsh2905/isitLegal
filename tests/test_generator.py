"""
Generator interface (generation/generator.py). The AnthropicClient and
PromptBuilder/generate() tests below use a scripted FAKE client -- a passing
test there is evidence the wiring around a generator is correct, not evidence
about what a real model would produce. The GroqClient tests at the bottom of
this file are a partial exception: a GROQ_API_KEY is configured in this
environment (see generation/generator.py's GroqClient docstring for how it
was actually exercised against a live model), but this suite still only
tests GroqClient's CONFIGURATION behaviour offline, never making a live call
itself -- CLAUDE.md invariant 6 requires every number to be reproducible by
running a script without a paid credential or network access.
"""
import os
import sys
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.generator import (
    PromptBuilder, RetrievedChunk, generate, SYSTEM_PROMPT,
    AnthropicClient, GeneratorNotConfigured)


from conftest import FakeClient


def test_prompt_includes_every_retrieved_chunk_with_its_index():
    chunks = [RetrievedChunk(1, "IPC§420", "cheating text"),
             RetrievedChunk(2, "CONST§21", "life and liberty text")]
    prompt = PromptBuilder.build("What is cheating?", chunks)
    assert "[1] IPC§420: cheating text" in prompt
    assert "[2] CONST§21: life and liberty text" in prompt


def test_prompt_contains_the_query():
    prompt = PromptBuilder.build("What is the punishment for murder?", [])
    assert "What is the punishment for murder?" in prompt


def test_system_prompt_forbids_outside_knowledge_and_requires_markers():
    assert "ONLY" in SYSTEM_PROMPT
    assert "[n]" in SYSTEM_PROMPT
    assert "quotation marks" in SYSTEM_PROMPT


def test_generate_calls_the_client_with_the_built_prompt():
    client = FakeClient("Cheating is punished under Section 420 IPC [1].")
    chunks = [RetrievedChunk(1, "IPC§420", "cheating text")]
    result = generate("What is cheating?", chunks, client)
    assert result.text == "Cheating is punished under Section 420 IPC [1]."
    assert len(client.calls) == 1
    system, user = client.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[1] IPC§420" in user


def test_generate_strips_the_response():
    client = FakeClient("  padded response  \n")
    result = generate("q", [], client)
    assert result.text == "padded response"


def test_anthropic_client_fails_loudly_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GeneratorNotConfigured):
        AnthropicClient()


def test_anthropic_client_does_not_fabricate_a_response_when_unconfigured(monkeypatch):
    # The failure mode under test: an unconfigured generator must raise, not
    # return an empty or plausible-looking string that downstream code could
    # mistake for real output.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        AnthropicClient()
        assert False, "should have raised"
    except GeneratorNotConfigured as e:
        assert "ANTHROPIC_API_KEY" in str(e)


# --------------------------------------------------------------------------
# GroqClient (generation/generator.py's second real LLMClient implementation)
#
# These test CONFIGURATION behaviour only, offline, with no key present --
# matching the AnthropicClient tests above and this project's reproducibility
# rule (CLAUDE.md invariant 6: every number must be reproducible by running a
# script, which is only true if the suite runs without network access or a
# paid credential). A live call against a real key exists but is deliberately
# NOT part of this suite: see generation/generator.py's GroqClient docstring
# for how that call was actually exercised, and why the default model
# (`openai/gpt-oss-20b`) was chosen empirically rather than assumed --
# a larger available model substituted its own citation-annotation format for
# the instructed `[n]` markers on a real test call, which would have passed
# every check here not because it verified correctly but because the check
# never saw a citation to verify.
# --------------------------------------------------------------------------

@contextmanager
def _clear_groq_key(monkeypatch):
    """
    Fully unconfigure GroqClient's three lookup paths (argument, environment,
    st.secrets), not just the environment variable. This repo's own
    .streamlit/secrets.toml genuinely carries a working key -- appropriate
    for the demo, wrong for a test claiming to exercise the unconfigured
    case, which would otherwise silently find that real key and pass for the
    wrong reason. st.secrets is a read-only object (attribute assignment
    raises TypeError), so the only reliable way to hide it from a fresh
    GroqClient() lookup is to move the file out of the way for the test and
    restore it afterwards, which is what this context manager does.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    secrets_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".streamlit", "secrets.toml")
    moved = secrets_path + ".hidden-for-test"
    existed = os.path.exists(secrets_path)
    if existed:
        os.rename(secrets_path, moved)
    try:
        yield
    finally:
        if existed:
            os.rename(moved, secrets_path)


def test_groq_client_fails_loudly_without_a_key(monkeypatch):
    from generation.generator import GroqClient
    with _clear_groq_key(monkeypatch):
        with pytest.raises(GeneratorNotConfigured):
            GroqClient()


def test_groq_client_does_not_fabricate_a_response_when_unconfigured(monkeypatch):
    from generation.generator import GroqClient
    with _clear_groq_key(monkeypatch):
        try:
            GroqClient()
            assert False, "should have raised"
        except GeneratorNotConfigured as e:
            assert "GROQ_API_KEY" in str(e)


def test_groq_client_accepts_an_explicit_key_argument(monkeypatch):
    # Construction must succeed on a key alone -- whether it resolves via
    # argument, environment, or st.secrets is app.py's concern, not this
    # class's; this only confirms the argument path itself is wired.
    from generation.generator import GroqClient
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = GroqClient(api_key="not-a-real-key-just-checking-construction")
    assert client.model == GroqClient.DEFAULT_MODEL


def test_groq_client_satisfies_the_llmclient_protocol():
    # Structural check: GroqClient must have the one method generate() calls,
    # with the right shape, independent of whether it's ever invoked live.
    from generation.generator import GroqClient
    import inspect
    sig = inspect.signature(GroqClient.complete)
    assert list(sig.parameters) == ["self", "system", "user"]
