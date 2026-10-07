"""Search across packs: keyword (FTS5 BM25), semantic (Int8 vectors) or hybrid (both).

Semantic scores are comparable across packs (one model, one vector space), so semantic results merge
by score. Keyword scores are not, so keyword and hybrid results merge by Reciprocal Rank Fusion: each
ranked list contributes 1 / (60 + rank) per item.
"""

from collections.abc import Iterable
from typing import Literal, Protocol

import numpy as np

from .packs import Hit, Pack, to_int8

Mode = Literal["keyword", "semantic", "hybrid"]
RRF_K = 60
INT8_SCALE = 127 * 127  # an Int8 dot product is cosine × 127²


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


def rrf(ranked_lists: Iterable[list[tuple[str, int]]]) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
    return scores


class Searcher:
    def __init__(self, packs: list[Pack], embedder: QueryEmbedder):
        self.packs = packs
        self.embedder = embedder

    def search(
        self,
        query: str,
        mode: Mode = "hybrid",
        limit: int = 10,
        kind: str | None = None,
        work_id: str | None = None,
    ) -> list[Hit]:
        packs = [
            p
            for p in self.packs
            if (kind is None or p.type == kind) and (work_id is None or p.work_id == work_id)
        ]
        if not packs:
            return []
        vector = None if mode == "keyword" else to_int8(self.embedder.embed_query(query))

        if mode == "semantic":
            scored = [(p, pos, s) for p in packs for pos, s in p.semantic(vector, limit)]
            scored.sort(key=lambda x: -x[2])
            return [p.hit(pos, s / INT8_SCALE) for p, pos, s in scored[:limit]]

        lists = []
        for p in packs:
            lists.append([(p.work_id, pos) for pos in p.keyword(query, limit * 2)])
            if mode == "hybrid":
                lists.append([(p.work_id, pos) for pos, _ in p.semantic(vector, limit * 2)])
        fused = sorted(rrf(lists).items(), key=lambda kv: -kv[1])[:limit]
        by_work = {p.work_id: p for p in packs}
        return [by_work[work].hit(pos, score) for (work, pos), score in fused]
