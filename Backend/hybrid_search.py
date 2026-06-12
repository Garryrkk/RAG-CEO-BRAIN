

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    METADATA = "metadata"
    RELATIONSHIP = "relationship"
    TIMELINE = "timeline"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryChunk:
    """
    A single retrievable unit of organisational memory.
    In production these come from your vector store / database.
    """
    chunk_id: str
    source_type: str          # email | meeting | document | commitment | risk …
    text: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    # metadata keys expected: date, company, project, person, tags, importance
    relationships: list[str] = field(default_factory=list)  # chunk_ids of related chunks
    timestamp: float = 0.0    # Unix epoch


@dataclass
class SearchResult:
    """A single search hit from any search mode."""
    chunk: MemoryChunk
    mode: SearchMode
    score: float              # 0–1, higher is better
    explanation: str = ""


@dataclass
class HybridSearchResult:
    """
    Merged, de-duplicated result produced by the HybridSearchEngine.
    Carries scores from all modes that found this chunk.
    """
    chunk: MemoryChunk
    final_score: float
    mode_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Individual search engines
# ---------------------------------------------------------------------------

class SemanticSearchEngine:
    """
    Cosine-similarity search over dense embeddings.
    In production: replace _get_embedding with a real model call
    and _cosine_sim with a vector-database ANN query.
    """

    def search(
        self,
        query: str,
        corpus: list[MemoryChunk],
        top_k: int = 10,
    ) -> list[SearchResult]:
        q_emb = self._get_embedding(query)
        scored = []
        for chunk in corpus:
            if chunk.embedding is None:
                chunk.embedding = self._get_embedding(chunk.text)
            score = self._cosine_sim(q_emb, chunk.embedding)
            scored.append(SearchResult(
                chunk=chunk,
                mode=SearchMode.SEMANTIC,
                score=score,
                explanation=f"Semantic similarity={score:.3f}",
            ))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Stub – replace with real embedding model
    # ------------------------------------------------------------------
    def _get_embedding(self, text: str) -> list[float]:
        """
        Stub: deterministic pseudo-embedding from character frequencies.
        Replace with: openai.Embedding.create / sentence-transformers / etc.
        """
        vec = [0.0] * 64
        for i, ch in enumerate(text[:128]):
            vec[i % 64] += ord(ch) / 1000.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
        norm_b = math.sqrt(sum(y * y for y in b)) or 1e-9
        return max(0.0, dot / (norm_a * norm_b))


class KeywordSearchEngine:
    """
    BM25-style keyword search for exact and partial term matching.
    """

    K1 = 1.5
    B  = 0.75

    def search(
        self,
        query: str,
        corpus: list[MemoryChunk],
        top_k: int = 10,
    ) -> list[SearchResult]:
        query_terms = self._tokenise(query)
        if not query_terms:
            return []

        # Build inverted index + doc lengths
        inv_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        doc_lengths: dict[str, int] = {}
        for chunk in corpus:
            tokens = self._tokenise(chunk.text)
            doc_lengths[chunk.chunk_id] = len(tokens)
            freq: dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            for term, count in freq.items():
                inv_index[term].append((chunk.chunk_id, count))

        avg_len = (
            sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1
        )
        N = len(corpus)
        chunk_map = {c.chunk_id: c for c in corpus}

        scores: dict[str, float] = defaultdict(float)
        for term in query_terms:
            postings = inv_index.get(term, [])
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for chunk_id, tf in postings:
                dl = doc_lengths[chunk_id]
                tf_norm = (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / avg_len)
                )
                scores[chunk_id] += idf * tf_norm

        # Normalise
        max_score = max(scores.values(), default=1.0) or 1.0
        results = []
        for chunk_id, raw in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            chunk = chunk_map[chunk_id]
            score = raw / max_score
            results.append(SearchResult(
                chunk=chunk,
                mode=SearchMode.KEYWORD,
                score=score,
                explanation=f"BM25={raw:.3f} (norm={score:.3f})",
            ))
        return results

    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9]+\b", text.lower())


class MetadataSearchEngine:
    """
    Structured filter-based search over metadata fields.
    Returns exact / partial matches on company, project, person, tags, date range.
    """

    def search(
        self,
        filters: dict,
        corpus: list[MemoryChunk],
        top_k: int = 10,
    ) -> list[SearchResult]:
        results = []
        for chunk in corpus:
            score = self._score_filters(filters, chunk.metadata)
            if score > 0:
                results.append(SearchResult(
                    chunk=chunk,
                    mode=SearchMode.METADATA,
                    score=score,
                    explanation=f"Metadata match score={score:.2f} for filters={filters}",
                ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _score_filters(self, filters: dict, metadata: dict) -> float:
        if not filters:
            return 0.0
        matched = 0
        for key, value in filters.items():
            meta_val = metadata.get(key, "")
            if isinstance(meta_val, list):
                if value in meta_val:
                    matched += 1
            elif isinstance(meta_val, str):
                if value.lower() in meta_val.lower():
                    matched += 1
        return matched / len(filters)


class RelationshipSearchEngine:
    """
    Graph-based traversal to find memory chunks connected to the
    entities mentioned in the query.
    """

    def search(
        self,
        entity_ids: list[str],
        corpus: list[MemoryChunk],
        depth: int = 2,
        top_k: int = 10,
    ) -> list[SearchResult]:
        chunk_map = {c.chunk_id: c for c in corpus}
        visited: set[str] = set()
        frontier: set[str] = set(entity_ids)
        all_found: dict[str, int] = {}   # chunk_id → hop distance

        for hop in range(depth):
            next_frontier: set[str] = set()
            for cid in frontier:
                if cid in visited:
                    continue
                visited.add(cid)
                if cid in chunk_map and cid not in entity_ids:
                    all_found[cid] = hop + 1
                chunk = chunk_map.get(cid)
                if chunk:
                    for rel in chunk.relationships:
                        if rel not in visited:
                            next_frontier.add(rel)
            frontier = next_frontier

        results = []
        for cid, hop in all_found.items():
            score = 1.0 / (1 + hop)   # closer = higher score
            results.append(SearchResult(
                chunk=chunk_map[cid],
                mode=SearchMode.RELATIONSHIP,
                score=score,
                explanation=f"Relationship hop={hop}, score={score:.2f}",
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class TimelineSearchEngine:
    """
    Returns memory chunks ordered by recency within a time window.
    Boosts recent items and allows date-range slicing.
    """

    def search(
        self,
        corpus: list[MemoryChunk],
        max_age_days: Optional[int] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        now = time.time()
        results = []
        for chunk in corpus:
            age_days = (now - chunk.timestamp) / 86400 if chunk.timestamp else 9999
            if max_age_days and age_days > max_age_days:
                continue
            # Recency score: decays over ~90 days
            score = math.exp(-age_days / 90.0)
            results.append(SearchResult(
                chunk=chunk,
                mode=SearchMode.TIMELINE,
                score=score,
                explanation=f"Age={age_days:.1f}d, recency_score={score:.3f}",
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """
    Executes all five search modes and fuses results using
    Reciprocal Rank Fusion (RRF) with configurable mode weights.

    Usage
    -----
    engine = HybridSearchEngine()
    results = engine.search(
        query="Smart Meter Deployment",
        corpus=memory_chunks,
        filters={"project": "Smart Meter"},
        entity_ids=["schneider_entity_001"],
        modes=["semantic","keyword","metadata","relationship","timeline"],
    )
    """

    # RRF constant (higher = less sensitive to rank differences)
    RRF_K = 60

    # Default mode weights (can be overridden per query)
    DEFAULT_WEIGHTS: dict[str, float] = {
        SearchMode.SEMANTIC:     0.35,
        SearchMode.KEYWORD:      0.25,
        SearchMode.METADATA:     0.20,
        SearchMode.RELATIONSHIP: 0.10,
        SearchMode.TIMELINE:     0.10,
    }

    def __init__(self) -> None:
        self._semantic    = SemanticSearchEngine()
        self._keyword     = KeywordSearchEngine()
        self._metadata    = MetadataSearchEngine()
        self._relationship = RelationshipSearchEngine()
        self._timeline    = TimelineSearchEngine()

    def search(
        self,
        query: str,
        corpus: list[MemoryChunk],
        filters: Optional[dict] = None,
        entity_ids: Optional[list[str]] = None,
        modes: Optional[list[str]] = None,
        max_age_days: Optional[int] = None,
        top_k: int = 20,
        weights: Optional[dict[str, float]] = None,
    ) -> list[HybridSearchResult]:

        active_modes = modes or [m.value for m in SearchMode]
        w = weights or self.DEFAULT_WEIGHTS

        all_results: list[SearchResult] = []

        if SearchMode.SEMANTIC.value in active_modes:
            all_results += self._semantic.search(query, corpus, top_k=top_k)

        if SearchMode.KEYWORD.value in active_modes:
            all_results += self._keyword.search(query, corpus, top_k=top_k)

        if SearchMode.METADATA.value in active_modes and filters:
            all_results += self._metadata.search(filters, corpus, top_k=top_k)

        if SearchMode.RELATIONSHIP.value in active_modes and entity_ids:
            all_results += self._relationship.search(entity_ids, corpus, top_k=top_k)

        if SearchMode.TIMELINE.value in active_modes:
            all_results += self._timeline.search(corpus, max_age_days, top_k=top_k)

        # Fuse using RRF
        return self._rrf_fusion(all_results, w, top_k)

    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        results: list[SearchResult],
        weights: dict,
        top_k: int,
    ) -> list[HybridSearchResult]:
        # Group by chunk_id
        by_chunk: dict[str, list[SearchResult]] = defaultdict(list)
        for r in results:
            by_chunk[r.chunk.chunk_id].append(r)

        # Per-mode ranked lists
        mode_ranked: dict[str, list[str]] = defaultdict(list)
        for r in results:
            mode_ranked[r.mode.value].append(r.chunk.chunk_id)

        # Compute RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        for mode, ranked_ids in mode_ranked.items():
            mode_weight = weights.get(mode, 1.0 / len(SearchMode))
            seen: dict[str, int] = {}
            for rank, cid in enumerate(ranked_ids):
                if cid not in seen:
                    seen[cid] = rank
            for cid, rank in seen.items():
                rrf_scores[cid] += mode_weight / (self.RRF_K + rank + 1)

        # Normalise
        max_rrf = max(rrf_scores.values(), default=1.0) or 1.0

        hybrid_results = []
        for cid, raw_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            hits = by_chunk[cid]
            mode_breakdown = {h.mode.value: round(h.score, 4) for h in hits}
            explanations = " | ".join(h.explanation for h in hits)
            hybrid_results.append(HybridSearchResult(
                chunk=hits[0].chunk,
                final_score=round(raw_score / max_rrf, 4),
                mode_scores=mode_breakdown,
                explanation=explanations,
            ))

        return hybrid_results


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    now = _time.time()

    sample_corpus = [
        MemoryChunk(
            chunk_id="email_001",
            source_type="email",
            text="Smart Meter Deployment approval is delayed pending regulator response.",
            metadata={"company": "Schneider Electric", "project": "Smart Meter", "tags": ["delay", "approval"]},
            relationships=["meeting_002", "commitment_003"],
            timestamp=now - 86400 * 2,
        ),
        MemoryChunk(
            chunk_id="meeting_002",
            source_type="meeting",
            text="KEI project team discussed Smart Meter rollout timeline. "
                 "Waiting for regulator response. Timeline shifted by 2 weeks.",
            metadata={"company": "KEI", "project": "Smart Meter", "tags": ["timeline", "risk"]},
            relationships=["email_001", "risk_004"],
            timestamp=now - 86400 * 5,
        ),
        MemoryChunk(
            chunk_id="commitment_003",
            source_type="commitment",
            text="John Smith committed to following up on regulator approval by March 25.",
            metadata={"person": "John Smith", "project": "Smart Meter", "status": "open", "tags": ["commitment"]},
            relationships=["email_001"],
            timestamp=now - 86400 * 3,
        ),
        MemoryChunk(
            chunk_id="risk_004",
            source_type="risk",
            text="Risk: Smart Meter deployment may slip Q2 if regulator approval delayed further.",
            metadata={"company": "Schneider Electric", "project": "Smart Meter", "tags": ["risk", "delay"]},
            relationships=["meeting_002"],
            timestamp=now - 86400 * 1,
        ),
        MemoryChunk(
            chunk_id="doc_005",
            source_type="document",
            text="Investor presentation covering Q1 highlights and Smart Meter progress.",
            metadata={"tags": ["investor", "presentation"], "project": "Smart Meter"},
            timestamp=now - 86400 * 30,
        ),
    ]

    engine = HybridSearchEngine()

    queries = [
        ("Smart Meter Deployment", {"project": "Smart Meter"}, ["meeting_002"]),
        ("What risks exist around deployment?", {"tags": "risk"}, []),
        ("John Smith commitment follow-up", {"person": "John Smith"}, ["commitment_003"]),
    ]

    print("=" * 70)
    print("PHASE 4 - TASK 3: HYBRID SEARCH ENGINE — DEMO")
    print("=" * 70)

    for query, filters, entity_ids in queries:
        print(f"\nQuery   : {query}")
        print(f"Filters : {filters}  |  Entities: {entity_ids}")
        results = engine.search(
            query=query,
            corpus=sample_corpus,
            filters=filters,
            entity_ids=entity_ids,
            top_k=5,
        )
        for i, r in enumerate(results, 1):
            print(f"  #{i}  [{r.chunk.source_type:12}] score={r.final_score:.4f}  "
                  f"modes={list(r.mode_scores.keys())}")
            print(f"       {r.chunk.text[:90]}")
