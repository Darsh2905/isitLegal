"""
CAPTURE (network, paid, non-deterministic): the second live generation round.

Not part of the regression harness. Run once; it writes the raw completions to
eval/live_round2_answers.json, and eval/run_live_round2_eval.py re-derives
everything else -- retrieval, gate decision, verification, audit -- from that
frozen text on every run (CLAUDE.md invariant 6). Resumable: an entry already
in the JSON file is not requested again.

WHY A SECOND ROUND. A 2026-10 peer review judged the first round too thin to
carry the headline numbers: ten answers from one model, on sixteen questions
this project wrote, with the pre/post improvement resting on one citation out
of 29. This round changes all three:

  queries  -- the 50 AILA 2019 queries (FIRE track, Bhattacharya et al.),
              fact situations written by the track organisers from Supreme
              Court judgments, not by this project; plus the 16 gold
              questions for continuity with round one.
  models   -- three independently invoked models, all that this account's
              Groq catalogue offers for chat: openai/gpt-oss-20b,
              openai/gpt-oss-120b, qwen/qwen3.8-27b.
  conditions -- GROUNDED (the deployed retrieval stack: stemmed BM25 +
              passage-window MiniLM, RRF, passage reranking, in-force
              preference, top 5) and UNGROUNDED (no retrieved context; the
              model is asked to cite provisions from its own knowledge --
              the configuration in which Bindra et al. measured 1.8%
              citation accuracy).

AS CAPTURED (2026-10-13). The run was stopped when the provider's daily token
quota throttled the gpt-oss models to about one answer per 25 minutes.
Queries are requested in a fixed order (gold q01-q16, then AILA_Q1 onwards),
so each model's answers cover a prefix of that order, not a selection:
gpt-oss-120b through AILA_Q24, gpt-oss-20b through AILA_Q23, qwen3.8-27b the
gold questions only (its 1,000 output-tokens-per-minute limit). 181 answers.
The frozen file records this under "stopped".

The grounded condition generates for EVERY query, recording the confidence
gate's decision rather than enforcing it, so the verifier is exercised on as
much real output as possible; run_live_round2_eval.py reports figures both
over all answers and over the answers the deployed gate would have let
through.

gpt-oss models are called with reasoning_effort="low": at the default
effort their hidden reasoning consumed the token budget and a third of
ungrounded answers came back cut off at max_tokens in a trial run. Truncation
is still recorded per answer (finish_reason) and reported.

Chunk text sent to the model is capped at CHUNK_WORDS words (marked
"[...]") so a request fits the provider's 8,000 tokens-per-minute limit; the
verifier checks against the full provision text, so a quotation from the
uncapped remainder is still checked against real law.
"""
import json
import os
import sys
import threading
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "live_round2_answers.json")
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK_WORDS = 350
MAX_TOKENS = 2048

UNGROUNDED_SYSTEM = """You are an expert on Indian law.

Answer the question. Support every legal claim by naming the specific statutory provision (act and section or article) or case it rests on. If you quote statutory text, enclose it in double quotation marks. If you are unsure of a provision, say so rather than guessing."""

AILA_PROMPT = ("On the following facts, which provisions of Indian law are "
               "relevant, and what do they provide?\n\nFacts: {facts}")


def load_queries():
    from eval.gold_qa import GOLD
    qs = [(f"gold-{qid}", "gold", q) for qid, q, _rel, _inf in GOLD]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "aila", "Query_doc.txt")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "||" not in line:
                continue
            qid, text = line.rstrip("\n").split("||", 1)
            qs.append((qid.strip(), "aila", AILA_PROMPT.format(facts=text.strip())))
    return qs


def retrieve_all(queries):
    """Top-5 chunk keys and gate decision per query (deterministic)."""
    from ingest.india_code import build_real_db
    from retrieval.bm25_baseline import BM25Retriever
    from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
    from retrieval.fusion import FusedRetriever
    from retrieval.rerank import Reranker
    from retrieval.inforce import prefer_in_force
    from retrieval.confidence import ConfidenceGate
    from eval.run_live_round2_eval import retrieval_stack, gate_for
    db = build_real_db()
    fused, reranker, gate = retrieval_stack(db), Reranker(db), gate_for(db)
    out = {}
    for qid, src, q in queries:
        cands = fused.search(q, 20)
        top = prefer_in_force(reranker.rerank(q, cands, top_k=5), db)
        ok, conf = gate.gate(q)
        out[qid] = {"chunk_keys": [k for k, _ in top], "gate_pass": bool(ok),
                    "gate_score": float(conf.score)}
    return db, out


def _cap(text):
    w = text.split()
    return text if len(w) <= CHUNK_WORDS else " ".join(w[:CHUNK_WORDS]) + " [...]"


def main():
    import tomllib
    import groq
    from generation.generator import SYSTEM_PROMPT
    key = os.environ.get("GROQ_API_KEY") or tomllib.load(
        open(os.path.join(os.path.dirname(OUT), "..", ".streamlit",
                          "secrets.toml"), "rb"))["GROQ_API_KEY"]
    # timeout: a stalled connection otherwise blocked one worker for the
    # client's 600 s default per attempt (seen in a trial run).
    client = groq.Groq(api_key=key, max_retries=0, timeout=120)

    queries = load_queries()
    data = json.load(open(OUT)) if os.path.exists(OUT) else {
        "captured": str(date.today()), "models": MODELS,
        "chunk_words": CHUNK_WORDS, "max_tokens": MAX_TOKENS,
        "gpt_oss_reasoning_effort": "low",
        "retrieval": {}, "answers": {}}
    print("retrieving...", flush=True)
    db, retr = retrieve_all(queries)
    data["retrieval"] = retr
    # Read every chunk text once, here: the SQLite connection is not safe to
    # share across the worker threads (a concurrent read returned None and
    # killed one worker in a trial run).
    texts = {ck: _cap(db.get(ck).text) for r in retr.values()
             for ck in r["chunk_keys"]}
    lock = threading.Lock()

    def save():
        with lock:
            tmp = OUT + ".tmp"
            json.dump(data, open(tmp, "w"), indent=1, ensure_ascii=False)
            os.replace(tmp, OUT)

    def call(model, system, user, max_tokens=MAX_TOKENS):
        for attempt in range(12):
            try:
                extra = ({"reasoning_effort": "low"}
                         if model.startswith("openai/gpt-oss") else {})
                r = client.chat.completions.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}], **extra)
                fin = r.choices[0].finish_reason
                if max_tokens != MAX_TOKENS:
                    fin = f"{fin}@{max_tokens}"
                return (r.choices[0].message.content or "", fin)
            except groq.RateLimitError as e:
                if "Request too large" in str(e):
                    # Prompt + max_tokens exceeds a per-minute cap: no
                    # amount of waiting helps. One retry with a smaller
                    # budget (recorded in finish_reason), then give up.
                    print(f"  too large {model}: {str(e)[:160]}", flush=True)
                    if max_tokens > 1000:
                        return call(model, system, user, max_tokens=1000)
                    return None, "too_large"
                wait = float(getattr(e, "response", None) and
                             e.response.headers.get("retry-after") or 20)
                print(f"  429 {model} retry in {wait}s: {str(e)[:120]}", flush=True)
                time.sleep(min(max(wait, 5), 90))
            except groq.APIStatusError as e:
                print(f"  {e.status_code} {model}: {str(e)[:120]}", flush=True)
                if e.status_code == 413:      # request larger than the TPM cap
                    return None, "too_large"
                time.sleep(15)
            except Exception as e:
                print(f"  {type(e).__name__} {model}: {str(e)[:120]}", flush=True)
                time.sleep(15)
        return None, "failed"

    # qwen3.8-27b on this account is limited to 1,000 OUTPUT tokens per
    # minute; a request asking for more than that is refused as "too large"
    # (112 requests in a trial run). Its budget is therefore 1,000 tokens,
    # recorded in each answer's finish_reason ("stop@1000").
    budget = {"qwen/qwen3.8-27b": 1000}

    def worker(model):
        for qid, src, q in queries:
            for cond in ("grounded", "ungrounded"):
                k = f"{model}|{cond}|{qid}"
                if k in data["answers"] and data["answers"][k]["text"] is not None:
                    continue
                if cond == "grounded":
                    keys = retr[qid]["chunk_keys"]
                    lines = [f"Question: {q}", "", "Source chunks:"]
                    for i, ck in enumerate(keys, 1):
                        lines.append(f"[{i}] {ck}: {texts[ck]}")
                    lines += ["", "Answer, following the rules exactly."]
                    text, fin = call(model, SYSTEM_PROMPT, "\n".join(lines),
                                     budget.get(model, MAX_TOKENS))
                else:
                    text, fin = call(model, UNGROUNDED_SYSTEM, q,
                                     budget.get(model, MAX_TOKENS))
                with lock:
                    data["answers"][k] = {"model": model, "condition": cond,
                                          "qid": qid, "source": src,
                                          "text": text, "finish_reason": fin}
                save()
                print(f"{model:22} {cond:10} {qid:12} {fin}", flush=True)

    threads = [threading.Thread(target=worker, args=(m,)) for m in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    save()
    print("done", len(data["answers"]))


if __name__ == "__main__":
    main()
