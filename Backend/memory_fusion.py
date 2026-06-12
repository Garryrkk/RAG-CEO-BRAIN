from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FusedMemory:
    """
    A single unified memory object produced by fusing multiple related chunks.
    """
    fusion_id: str
    synthesised_text: str           # The fused, unified narrative
    source_chunks: list[MemoryChunk]
    topic: str                      # inferred topic label
    confidence: float               # 0-1
    conflicts_detected: bool = False
    conflict_notes: str = ""
    timestamp_range: tuple[float, float] = (0.0, 0.0)
    tags: list[str] = field(default_factory=list)

    def source_ids(self) -> list[str]:
        return [c.chunk_id for c in self.source_chunks]

    def date_range_str(self) -> str:
        if not self.timestamp_range[0]:
            return "unknown"
        start = datetime.fromtimestamp(self.timestamp_range[0]).strftime("%b %d, %Y")
        end   = datetime.fromtimestamp(self.timestamp_range[1]).strftime("%b %d, %Y")
        return f"{start} – {end}" if start != end else start


@dataclass
class FusionReport:
    """Summary of a fusion operation."""
    input_chunk_count: int
    output_fused_count: int
    conflicts_found: int
    fused_memories: list[FusedMemory]
    ungrouped_chunks: list[MemoryChunk]  # chunks that didn't fuse with anything


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity – fast approximate overlap."""
    set_a = set(re.findall(r"\b[a-z0-9]+\b", text_a.lower()))
    set_b = set(re.findall(r"\b[a-z0-9]+\b", text_b.lower()))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _shared_entities(meta_a: dict, meta_b: dict) -> int:
    """Count how many metadata keys have matching values."""
    shared = 0
    for key in ("company", "project", "person"):
        val_a = meta_a.get(key, "")
        val_b = meta_b.get(key, "")
        if val_a and val_b and val_a.lower() == val_b.lower():
            shared += 1
    return shared


def _temporal_proximity(ts_a: float, ts_b: float, window_days: float = 14.0) -> bool:
    """Return True if two timestamps are within window_days of each other."""
    if not ts_a or not ts_b:
        return True   # unknown dates — allow fusion
    return abs(ts_a - ts_b) <= window_days * 86400


def _should_fuse(chunk_a: MemoryChunk, chunk_b: MemoryChunk,
                 min_jaccard: float = 0.15,
                 min_shared_entities: int = 1) -> bool:
    """Decide whether two chunks describe the same situation."""
    entity_match = _shared_entities(chunk_a.metadata, chunk_b.metadata)
    if entity_match < min_shared_entities:
        return False
    text_sim = _jaccard_similarity(chunk_a.text, chunk_b.text)
    temporal = _temporal_proximity(chunk_a.timestamp, chunk_b.timestamp)
    # Relationship link is a hard signal
    rel_linked = (chunk_b.chunk_id in chunk_a.relationships or
                  chunk_a.chunk_id in chunk_b.relationships)
    return rel_linked or (text_sim >= min_jaccard and temporal)


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

CONFLICT_PAIRS = [
    (r"\bapproved\b", r"\brejected\b"),
    (r"\bcompleted?\b", r"\bpending\b"),
    (r"\bon track\b", r"\bdelayed\b"),
    (r"\bno risk\b", r"\brisk\b"),
    (r"\bresolv\w+\b", r"\bunresolv\w+\b"),
    (r"\bconfirmed\b", r"\buncertain\b"),
]


def _detect_conflicts(texts: list[str]) -> tuple[bool, str]:
    """
    Scan a list of texts for contradictory signals.
    Returns (conflict_found, description).
    """
    notes = []
    for pat_a, pat_b in CONFLICT_PAIRS:
        sources_with_a = [t for t in texts if re.search(pat_a, t, re.I)]
        sources_with_b = [t for t in texts if re.search(pat_b, t, re.I)]
        if sources_with_a and sources_with_b:
            notes.append(
                f"Conflict: '{pat_a}' vs '{pat_b}' found across sources."
            )
    return (bool(notes), "; ".join(notes))


# ---------------------------------------------------------------------------
# Text synthesis
# ---------------------------------------------------------------------------

def _synthesise_texts(chunks: list[MemoryChunk]) -> str:
    """
    Rule-based synthesis: deduplicates sentences and builds a coherent paragraph.
    In production: replace with an LLM call for richer synthesis.
    """
    # Collect all sentences
    all_sentences: list[str] = []
    for chunk in sorted(chunks, key=lambda c: c.timestamp):
        for sent in re.split(r"(?<=[.!?])\s+", chunk.text.strip()):
            sent = sent.strip()
            if len(sent) > 10:
                all_sentences.append(sent)

    # De-duplicate near-identical sentences (Jaccard > 0.6)
    unique: list[str] = []
    for candidate in all_sentences:
        is_duplicate = any(
            _jaccard_similarity(candidate, existing) > 0.60
            for existing in unique
        )
        if not is_duplicate:
            unique.append(candidate)

    return " ".join(unique)


def _infer_topic(chunks: list[MemoryChunk]) -> str:
    """Infer the most likely topic label from shared metadata."""
    project  = next((c.metadata.get("project")  for c in chunks if c.metadata.get("project")),  "")
    company  = next((c.metadata.get("company")  for c in chunks if c.metadata.get("company")),  "")
    person   = next((c.metadata.get("person")   for c in chunks if c.metadata.get("person")),   "")
    if project:
        return f"Project: {project}"
    if company:
        return f"Company: {company}"
    if person:
        return f"Person: {person}"
    return "General"


def _collect_tags(chunks: list[MemoryChunk]) -> list[str]:
    all_tags: set[str] = set()
    for c in chunks:
        tags = c.metadata.get("tags", [])
        if isinstance(tags, list):
            all_tags.update(tags)
        elif isinstance(tags, str):
            all_tags.add(tags)
    return sorted(all_tags)


# ---------------------------------------------------------------------------
# Grouping (Union-Find)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)

    def groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


# ---------------------------------------------------------------------------
# Main fusion engine
# ---------------------------------------------------------------------------

class MemoryFusionEngine:
    """
    Groups related memory chunks and synthesises each group into a
    single FusedMemory object.

    Usage
    -----
    engine = MemoryFusionEngine()
    report = engine.fuse(chunks)
    for fm in report.fused_memories:
        print(fm.synthesised_text)
    """

    def __init__(
        self,
        min_jaccard: float = 0.15,
        min_shared_entities: int = 1,
    ) -> None:
        self._min_jaccard = min_jaccard
        self._min_shared_entities = min_shared_entities

    def fuse(self, chunks: list[MemoryChunk]) -> FusionReport:
        if not chunks:
            return FusionReport(0, 0, 0, [], [])

        n = len(chunks)
        uf = _UnionFind(n)

        # Pairwise fusion decisions
        for i in range(n):
            for j in range(i + 1, n):
                if _should_fuse(chunks[i], chunks[j],
                                self._min_jaccard,
                                self._min_shared_entities):
                    uf.union(i, j)

        groups = uf.groups()

        fused_memories: list[FusedMemory] = []
        ungrouped: list[MemoryChunk] = []
        conflicts_total = 0

        for root, indices in groups.items():
            group_chunks = [chunks[i] for i in indices]

            if len(group_chunks) == 1:
                ungrouped.append(group_chunks[0])
                continue

            texts = [c.text for c in group_chunks]
            conflict_found, conflict_notes = _detect_conflicts(texts)
            if conflict_found:
                conflicts_total += 1

            synthesised = _synthesise_texts(group_chunks)
            topic = _infer_topic(group_chunks)
            tags  = _collect_tags(group_chunks)
            timestamps = [c.timestamp for c in group_chunks if c.timestamp]
            ts_range = (min(timestamps), max(timestamps)) if timestamps else (0.0, 0.0)

            fused_memories.append(FusedMemory(
                fusion_id=f"fusion_{root:04d}_{int(time.time())}",
                synthesised_text=synthesised,
                source_chunks=group_chunks,
                topic=topic,
                confidence=round(1.0 - 0.1 * conflict_found, 2),
                conflicts_detected=conflict_found,
                conflict_notes=conflict_notes,
                timestamp_range=ts_range,
                tags=tags,
            ))

        return FusionReport(
            input_chunk_count=n,
            output_fused_count=len(fused_memories),
            conflicts_found=conflicts_total,
            fused_memories=fused_memories,
            ungrouped_chunks=ungrouped,
        )

    def fuse_from_results(
        self, results: list[HybridSearchResult]
    ) -> FusionReport:
        chunks = [r.chunk for r in results]
        return self.fuse(chunks)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    now = _time.time()

    chunks = [
        MemoryChunk(
            chunk_id="email_001",
            source_type="email",
            text="Approval delayed. Waiting for regulator sign-off on Smart Meter deployment.",
            metadata={"company": "Schneider Electric", "project": "Smart Meter", "tags": ["delay"]},
            relationships=["meeting_002"],
            timestamp=now - 86400 * 3,
        ),
        MemoryChunk(
            chunk_id="meeting_002",
            source_type="meeting",
            text="Waiting for regulator response. Smart Meter rollout is on hold.",
            metadata={"company": "Schneider Electric", "project": "Smart Meter", "tags": ["delay"]},
            relationships=["email_001", "project_003"],
            timestamp=now - 86400 * 5,
        ),
        MemoryChunk(
            chunk_id="project_003",
            source_type="project",
            text="Smart Meter project timeline shifted by 2 weeks due to approval delay.",
            metadata={"company": "Schneider Electric", "project": "Smart Meter"},
            relationships=["meeting_002"],
            timestamp=now - 86400 * 2,
        ),
        MemoryChunk(
            chunk_id="investor_004",
            source_type="email",
            text="Investor call went well. Discussed Q1 results and pipeline.",
            metadata={"tags": ["investor"]},
            timestamp=now - 86400 * 10,
        ),
    ]

    engine = MemoryFusionEngine()
    report = engine.fuse(chunks)

    print("=" * 70)
    print("PHASE 4 - TASK 5: MEMORY FUSION — DEMO")
    print("=" * 70)
    print(f"Input chunks  : {report.input_chunk_count}")
    print(f"Fused memories: {report.output_fused_count}")
    print(f"Conflicts     : {report.conflicts_found}")
    print(f"Ungrouped     : {len(report.ungrouped_chunks)}")

    for fm in report.fused_memories:
        print(f"\n--- Fused Memory: {fm.fusion_id} ---")
        print(f"  Topic     : {fm.topic}")
        print(f"  Sources   : {fm.source_ids()}")
        print(f"  Date range: {fm.date_range_str()}")
        print(f"  Conflicts : {fm.conflicts_detected} {fm.conflict_notes or ''}")
        print(f"  Confidence: {fm.confidence}")
        print(f"  Synthesis :")
        print(f"    {fm.synthesised_text}")

    print("\n--- Ungrouped (standalone) ---")
    for c in report.ungrouped_chunks:
        print(f"  [{c.chunk_id}] {c.text[:80]}")
