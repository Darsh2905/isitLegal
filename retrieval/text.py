"""Shared lexical tokenisation for BM25 retrieval.

Stemming. Unstemmed BM25 treats "arrested"/"arrest", "cheating"/"cheats",
"punished"/"punishment" as unrelated terms, so a natural-language question
phrased with a different inflection than the statute's own drafting misses the
provision entirely. Measured on the real corpus (16 gold queries,
eval/run_real_corpus_eval.py reports both arms): Recall@1 0.375 -> 0.500,
Recall@5 0.688 -> 0.813, MRR@10 0.525 -> 0.609, at the cost of Recall@10
0.875 -> 0.813 (one query falls out of the top 10). The standard Snowball
(Porter2) stemmer is used rather than a hand-tuned one: a scratch suffix-
stripper scored higher on these same 16 queries (Recall@1 0.563), and picking
the stemmer by its score on the evaluation set would be tuning on the test
set. Deterministic: no model, no learned weights.
"""
import re
from functools import lru_cache

import snowballstemmer

_TOK = re.compile(r"[a-z0-9]+")
_STEMMER = snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return word if word.isdigit() else _STEMMER.stemWord(word)


def tokenize(text: str, stem: bool = True):
    toks = _TOK.findall((text or "").lower())
    return [_stem(t) for t in toks] if stem else toks
