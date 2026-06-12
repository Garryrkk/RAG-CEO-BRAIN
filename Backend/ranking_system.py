
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Authority tiers – higher = more authoritative
SOURCE_AUTHORITY: dict[str, float] = {
    "ceo_email":      1.0,
    "board_note":     0.95,
    "email":          0.80,
    "meeting":        0.75,
    "commitment":     0.85,
    "risk":           0.85,
    "project":        0.70,
    "document":       0.65,
    "conversation":   0.60,
    "stakeholder":    0.60,
    "timeline":       0.55,
    "attachment":     0.40,
    "unknown":        0.50,
}

# Importance signal keywords found in text or metadata
HIGH_IMPORTANCE_KEYWORDS = [
    "ceo", "board", "urgent", "critical", "escalat",
    "deadline", "overdue", "risk", "blocked", "investor",
    "regulator", "compliance", "contract", "signed",
]

MEDIUM_IMPORTANCE_KEYWORDS = [
    "follow.up", "commitment", "project", "client",
    "pending", "approval", "budget", "milestone",
]


# ---------------------------------------------------------------------------
# Scoring sub-functions
# ---------------------------------------------------------------------------

def _relevance_score(result: HybridSearchResult) -> float:
    """Use the hybrid search final score as the relevance signal."""
    return max(0.0, min(1.0, result.final_score))


def _freshness_score(timestamp: float, half_life_days: float = 90.0) -> float:
    """
    Exponential decay. Information loses half its freshness score every
    half_life_days days. Very recent items score close to 1.0.
    """
    if not timestamp:
        return 0.5  # unknown – neutral
    age_days = (time.time() - timestamp) / 86400
    return math.exp(-age_days * math.log(2) / half_life_days)


def _authority_score(source_type: str, metadata: dict) -> float:
    """
    Look up the source type in the authority table.
    Boost if metadata signals a senior sender.
    """
    base = SOURCE_AUTHORITY.get(source_type.lower(), SOURCE_AUTHORITY["unknown"])
    sender = str(metadata.get("sender", "") or metadata.get("person", "")).lower()
    if any(kw in sender for kw in ("ceo", "founder", "chairman", "president")):
        base = min(base + 0.15, 1.0)
    return base


def _importance_score(text: str, metadata: dict) -> float:
    """
    Keyword-based importance detection.
    """
    combined = (text + " " + str(metadata)).lower()
    high = sum(1 for kw in HIGH_IMPORTANCE_KEYWORDS if kw in combined)
    medium = sum(1 for kw in MEDIUM_IMPORTANCE_KEYWORDS if kw in combined)
    raw = (high * 2 + medium * 1) / (len(HIGH_IMPORTANCE_KEYWORDS) * 2)
    return min(raw, 1.0)


def _relationship_strength_score(
    chunk: MemoryChunk,
    focus_entity_ids: Optional[list[str]] = None,
) -> float:
    """
    Score based on how directly the chunk is linked to the query's focus entity.
    Direct relationship = 1.0, two hops = 0.5, no link = 0.1.
    """
    if not focus_entity_ids:
        return 0.5  # unknown – neutral

    direct = any(eid in chunk.relationships for eid in focus_entity_ids)
    if direct:
        return 1.0

    # Is the entity mentioned in metadata?
    meta_vals = " ".join(str(v) for v in chunk.metadata.values()).lower()
    entity_mention = any(eid.lower().replace("_", " ") in meta_vals
                         for eid in focus_entity_ids)
    if entity_mention:
        return 0.7

    return 0.2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredResult:
    """A HybridSearchResult with a full ranking scorecard."""
    result: HybridSearchResult
    relevance:     float = 0.0
    freshness:     float = 0.0
    authority:     float = 0.0
    importance:    float = 0.0
    relationship:  float = 0.0
    final_rank_score: float = 0.0

    def breakdown(self) -> str:
        return (
            f"rank={self.final_rank_score:.4f}  "
            f"rel={self.relevance:.2f}  "
            f"fresh={self.freshness:.2f}  "
            f"auth={self.authority:.2f}  "
            f"imp={self.importance:.2f}  "
            f"relstr={self.relationship:.2f}"
        )


@dataclass
class RankingWeights:
    """Tunable weights for each ranking factor (must sum to 1.0)."""
    relevance:    float = 0.35
    freshness:    float = 0.25
    authority:    float = 0.15
    importance:   float = 0.15
    relationship: float = 0.10

    def __post_init__(self) -> None:
        total = (self.relevance + self.freshness + self.authority +
                 self.importance + self.relationship)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"RankingWeights must sum to 1.0, got {total:.2f}")


# ---------------------------------------------------------------------------
# Preset weight profiles
# ---------------------------------------------------------------------------

PROFILE_EXECUTIVE_QUERY = RankingWeights(
    relevance=0.35, freshness=0.30, authority=0.15, importance=0.15, relationship=0.05
)

PROFILE_RISK_QUERY = RankingWeights(
    relevance=0.25, freshness=0.25, authority=0.10, importance=0.30, relationship=0.10
)

PROFILE_COMMITMENT_QUERY = RankingWeights(
    relevance=0.25, freshness=0.20, authority=0.10, importance=0.20, relationship=0.25
)

PROFILE_HISTORICAL = RankingWeights(
    relevance=0.40, freshness=0.10, authority=0.20, importance=0.20, relationship=0.10
)


# ---------------------------------------------------------------------------
# Main ranking engine
# ---------------------------------------------------------------------------

class RankingEngine:
    """
    Scores and re-ranks a list of HybridSearchResults using a multi-factor
    scoring model.

    Usage
    -----
    engine = RankingEngine()
    ranked = engine.rank(results, focus_entity_ids=["schneider_001"])
    for sr in ranked:
        print(sr.result.chunk.chunk_id, sr.breakdown())
    """

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        half_life_days: float = 90.0,
    ) -> None:
        self._weights = weights or PROFILE_EXECUTIVE_QUERY
        self._half_life = half_life_days

    def rank(
        self,
        results: list[HybridSearchResult],
        focus_entity_ids: Optional[list[str]] = None,
    ) -> list[ScoredResult]:

        scored: list[ScoredResult] = []

        for r in results:
            chunk = r.chunk
            w = self._weights

            rel   = _relevance_score(r)
            fresh = _freshness_score(chunk.timestamp, self._half_life)
            auth  = _authority_score(chunk.source_type, chunk.metadata)
            imp   = _importance_score(chunk.text, chunk.metadata)
            relstr = _relationship_strength_score(chunk, focus_entity_ids)

            final = (
                w.relevance    * rel   +
                w.freshness    * fresh +
                w.authority    * auth  +
                w.importance   * imp   +
                w.relationship * relstr
            )

            scored.append(ScoredResult(
                result=r,
                relevance=rel,
                freshness=fresh,
                authority=auth,
                importance=imp,
                relationship=relstr,
                final_rank_score=round(final, 4),
            ))

        scored.sort(key=lambda s: s.final_rank_score, reverse=True)
        return scored

    def rank_and_filter(
        self,
        results: list[HybridSearchResult],
        top_k: int = 10,
        min_score: float = 0.0,
        focus_entity_ids: Optional[list[str]] = None,
    ) -> list[ScoredResult]:
        ranked = self.rank(results, focus_entity_ids)
        return [r for r in ranked if r.final_rank_score >= min_score][:top_k]

    def with_weights(self, weights: RankingWeights) -> "RankingEngine":
        """Return a new engine with different weights."""
        return RankingEngine(weights=weights, half_life_days=self._half_life)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_ranking_engine(query_category: str) -> RankingEngine:
    """
    Return a pre-configured RankingEngine for a given query category.
    """
    profiles = {
        "risk":       PROFILE_RISK_QUERY,
        "commitment": PROFILE_COMMITMENT_QUERY,
        "general":    PROFILE_EXECUTIVE_QUERY,
        "company":    PROFILE_EXECUTIVE_QUERY,
        "person":     PROFILE_EXECUTIVE_QUERY,
        "project":    PROFILE_EXECUTIVE_QUERY,
    }
    weights = profiles.get(query_category.lower(), PROFILE_EXECUTIVE_QUERY)
    return RankingEngine(weights=weights)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from task3_hybrid_search.hybrid_search import HybridSearchEngine

    now = _time.time()

    from task3_hybrid_search.hybrid_search import MemoryChunk

    corpus = [
        MemoryChunk("ceo_email_001", "ceo_email",
                    "URGENT: Smart Meter regulator approval is critical for Q2 delivery.",
                    metadata={"company": "Schneider Electric", "sender": "CEO John Smith",
                              "tags": ["urgent", "ceo"]},
                    relationships=["project_003"],
                    timestamp=now - 86400 * 1),
        MemoryChunk("old_attachment_002", "attachment",
                    "Smart Meter technical specifications v1.0",
                    metadata={"project": "Smart Meter"},
                    timestamp=now - 86400 * 1095),  # 3 years old
        MemoryChunk("meeting_003", "meeting",
                    "Smart Meter deployment timeline shifted due to regulator delay.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    relationships=["ceo_email_001"],
                    timestamp=now - 86400 * 3),
        MemoryChunk("risk_004", "risk",
                    "Risk: Regulatory approval overdue. Q2 at risk.",
                    metadata={"company": "Schneider Electric", "tags": ["risk", "overdue"]},
                    timestamp=now - 86400 * 2),
    ]

    search_engine = HybridSearchEngine()
    results = search_engine.search(
        query="Smart Meter deployment status",
        corpus=corpus,
        filters={"company": "Schneider Electric"},
        top_k=10,
    )

    ranking_engine = RankingEngine()
    ranked = ranking_engine.rank(results, focus_entity_ids=["ceo_email_001"])

    print("=" * 70)
    print("PHASE 4 - TASK 7: RANKING SYSTEM — DEMO")
    print("=" * 70)
    print(f"\n{'#':<4} {'Chunk ID':<25} {'Type':<15} {''}")
    for i, sr in enumerate(ranked, 1):
        print(f"  #{i:<3} {sr.result.chunk.chunk_id:<25} "
              f"{sr.result.chunk.source_type:<15} {sr.breakdown()}")
    print()
    print("Note: CEO email (1 day old) ranks above 3-year-old attachment.")
