"""
In-force preference: a deterministic post-retrieval reordering (contribution
C2 applied at retrieval time, not only at verification time).

A repealed provision and its successor share their marginal note and much of
their text ("Punishment for murder": IPC 302 and BNS 103), so a lexical or
embedding retriever has no signal separating them and ranks them close to
arbitrarily. On the real corpus BM25 ranks the in-force provision above its
repealed predecessor in only 4 of 9 gold pairs. The verifier already
annotates a repealed citation with its successor after the fact; this step
acts earlier, so the generator is handed current law first.

Rule: whenever a repealed provision and one of its recorded successors both
appear in a ranked list, the successor is moved to just ahead of the repealed
provision's position. Nothing is added or removed -- only order changes -- and
the rule uses nothing but the verified correspondence mapping, so it cannot
introduce a provision retrieval did not find.
"""
from typing import List, Tuple

from ingest.metadata_db import MetadataDB


def prefer_in_force(ranked: List[Tuple[str, float]],
                    db: MetadataDB) -> List[Tuple[str, float]]:
    out = list(ranked)
    for key, _ in list(ranked):
        for succ in db.successor(key):
            pos = {k: i for i, (k, _) in enumerate(out)}
            if succ in pos and key in pos and pos[succ] > pos[key]:
                item = out.pop(pos[succ])
                out.insert(pos[key], item)
    return out
