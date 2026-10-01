"""
Statutory cross-reference resolution and citation graph (contribution C5).

MOTIVATION
----------
Rustamov et al. (AICT 2025) diagnose a failure mode they leave unsolved: some
statutory provisions are not semantically self-contained. Their text merely
refers to other provisions ("punishment as provided in section X"), so it
carries almost no topical signal of its own. A query about the *substance* of
the offence therefore cannot reach them -- not by lexical match, because the
descriptive words live in the referenced section, and not by embedding
similarity, for the same reason. The authors propose a knowledge graph as
future work.

This module builds that graph. Two ideas, both deterministic:

1.  REFERENCE RESOLUTION. At index time we parse every outbound reference in a
    provision's text and store it as a typed edge. Intra-act references
    ("section 302" inside an IPC provision) are resolved against the enclosing
    act, which is what makes them recoverable at all -- the bare string
    "section 302" is ambiguous outside its statutory context.

2.  RETRIEVAL EXPANSION. After first-stage retrieval, we walk the graph
    outward by a bounded number of hops and admit neighbours the retriever
    could not have reached. Crucially the walk is INBOUND: given a query that
    matches section X, we surface provisions that *refer to* X, since those are
    exactly the ones whose own text is too thin to be retrieved directly.

The graph is also what makes the verifier's transitive checks possible: a
citation to a provision whose operative content lives elsewhere can be
annotated with the referenced provision.
"""

import re
import sys
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import MetadataDB
from ingest.normalize import extract_citations, extract_unresolved_act_refs, ACT_ALIASES


# A bare intra-act reference: "section 302", "sections 302 and 420",
# "section 138", "s. 173". No act name -- the act is inherited from context.
_BARE_SECTION = re.compile(
    r"\b(?:sections?|ss\.?|s\.)\s*"
    r"(?P<num>\d+[A-Za-z]?(?![A-Za-z])(?:\s*\([0-9a-zA-Z]+\))*)",
    re.IGNORECASE,
)

# Phrases signalling that the provision's operative content is elsewhere.
_DEPENDENCY_CUES = [
    "as provided in", "provided in", "specified in", "referred to in",
    "under section", "punishable under", "by reference to",
    "in accordance with", "save as otherwise provided",
    "contrary to", "except upon",
]


@dataclass(frozen=True)
class Edge:
    src: str            # provision key that contains the reference
    dst: str            # provision key referred to
    kind: str           # "explicit" (act named) | "intra" (act inherited)
    context: str        # the surrounding phrase, for auditability


class CitationGraph:
    """Directed graph over statutory provisions."""

    def __init__(self):
        self.out: Dict[str, Set[str]] = defaultdict(set)
        self.inn: Dict[str, Set[str]] = defaultdict(set)
        self.edges: List[Edge] = []
        self._edge_index: Dict[Tuple[str, str], Edge] = {}

    def add(self, e: Edge):
        if (e.src, e.dst) in self._edge_index or e.src == e.dst:
            return
        self.out[e.src].add(e.dst)
        self.inn[e.dst].add(e.src)
        self.edges.append(e)
        self._edge_index[(e.src, e.dst)] = e

    def outbound(self, key: str) -> Set[str]:
        return self.out.get(key, set())

    def inbound(self, key: str) -> Set[str]:
        return self.inn.get(key, set())

    def edge(self, src: str, dst: str) -> Optional[Edge]:
        return self._edge_index.get((src, dst))

    def expand(self, seeds: List[str], hops: int = 1,
               direction: str = "in") -> List[Tuple[str, int, str]]:
        """
        Bounded traversal from seed provisions.

        direction="in"   -> provisions that REFER TO the seeds. This is the
                            direction that recovers non-self-contained
                            provisions, which is the point of the exercise.
        direction="out"  -> provisions the seeds refer to (useful for supplying
                            the generator with referenced definitions).
        direction="both" -> union.

        Returns (key, hop_distance, via_seed), nearest first, excluding seeds.
        """
        seen = {s: 0 for s in seeds}
        origin = {s: s for s in seeds}
        q = deque((s, 0) for s in seeds)
        out: List[Tuple[str, int, str]] = []

        while q:
            node, d = q.popleft()
            if d >= hops:
                continue
            nbrs: Set[str] = set()
            if direction in ("in", "both"):
                nbrs |= self.inbound(node)
            if direction in ("out", "both"):
                nbrs |= self.outbound(node)
            for nb in nbrs:
                if nb in seen:
                    continue
                seen[nb] = d + 1
                origin[nb] = origin[node]
                out.append((nb, d + 1, origin[node]))
                q.append((nb, d + 1))

        out.sort(key=lambda t: t[1])
        return out

    def stats(self) -> dict:
        dangling = sum(1 for e in self.edges if e.dst is None)
        return {
            "nodes_with_edges": len(set(self.out) | set(self.inn)),
            "edges": len(self.edges),
            "explicit": sum(1 for e in self.edges if e.kind == "explicit"),
            "intra": sum(1 for e in self.edges if e.kind == "intra"),
        }


def _context_window(text: str, start: int, end: int, w: int = 45) -> str:
    return re.sub(r"\s+", " ",
                  text[max(0, start - w): min(len(text), end + w)]).strip()


def extract_references(key: str, act: str, text: str,
                       db: MetadataDB) -> List[Edge]:
    """
    Extract outbound references from one provision's text.

    Explicit references (act named) are resolved by the normaliser. Bare
    references ("section 302") inherit the enclosing act -- this inheritance is
    what makes them resolvable, and it is why resolution must happen at index
    time, where the enclosing act is known, rather than at query time.
    """
    edges: List[Edge] = []
    claimed: List[Tuple[int, int]] = []

    # 1. Explicit, act-qualified references. A sub-section pinpoint
    #    ("section 41(1)") resolves to the section the corpus indexes, by the
    #    same rule V2 uses (MetadataDB.resolve) -- formerly such edges were
    #    silently dropped as "nonexistent".
    for cit in extract_citations(text):
        dst = db.resolve_key(cit.key)
        if dst is None or dst == key:
            continue
        edges.append(Edge(key, dst, "explicit",
                          _context_window(text, cit.start, cit.end)))
        claimed.append((cit.start, cit.end))

    # References to an act OUTSIDE the corpus ("section 9 of the Code of Civil
    # Procedure") must not fall through to step 2 and inherit the enclosing
    # act -- that invents an edge to an unrelated section of this act.
    for ref in extract_unresolved_act_refs(text):
        claimed.append((ref.start, ref.end))

    # 2. Bare references, act inherited from the containing provision.
    for m in _BARE_SECTION.finditer(text):
        s, e = m.span()
        if any(not (e <= cs or s >= ce) for cs, ce in claimed):
            continue
        num = re.sub(r"\s+", "", m.group("num"))
        dst = db.resolve_key(f"{act}§{num}")
        if dst is None or dst == key:
            continue
        edges.append(Edge(key, dst, "intra", _context_window(text, s, e)))

    return edges


def build_graph(db: MetadataDB) -> CitationGraph:
    g = CitationGraph()
    for p in db.all_provisions():
        for e in extract_references(p.key, p.act, p.text or "", db):
            g.add(e)
    return g


def dependency_score(text: str) -> float:
    """
    Heuristic: how far does this provision lean on other provisions for its
    meaning? Used to flag non-self-contained sections at index time so they can
    be given graph-based reachability they would not otherwise have.

    Returns roughly [0, 1]; higher means more dependent.
    """
    if not text:
        return 0.0
    low = text.lower()
    cues = sum(1 for c in _DEPENDENCY_CUES if c in low)
    refs = len(_BARE_SECTION.findall(text)) + len(extract_citations(text))
    words = max(len(low.split()), 1)
    # Reference density, saturating, plus a cue bonus.
    return min(1.0, (refs / (words / 25.0)) * 0.5 + min(cues, 3) * 0.17)
