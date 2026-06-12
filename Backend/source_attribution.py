from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchResult
from task5_memory_fusion.memory_fusion import FusedMemory


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Attribution:
    """
    A single attribution record for one piece of information.
    """
    attribution_id: str
    source_type: str            # email | meeting | document | commitment | risk
    source_date: str            # human-readable date
    source_date_ts: float       # unix epoch for sorting
    participants: list[str]     # people involved or mentioned
    chunk_id: str               # original chunk reference
    excerpt: str                # short original text snippet (≤ 150 chars)
    confidence: float           # 0-1
    url_or_path: str = ""       # if available

    def to_inline_citation(self) -> str:
        """Short inline citation, e.g. [Email · Mar 18 · J. Smith]"""
        parts = [self.source_type.capitalize(), self.source_date]
        if self.participants:
            parts.append(", ".join(self.participants[:2]))
        return f"[{' · '.join(parts)}]"

    def to_full_citation(self) -> str:
        """Full audit-ready citation block."""
        lines = [
            f"Source     : {self.source_type.upper()}",
            f"Date       : {self.source_date}",
        ]
        if self.participants:
            lines.append(f"Participants: {', '.join(self.participants)}")
        lines.append(f"Reference  : {self.chunk_id}")
        if self.excerpt:
            lines.append(f'Excerpt    : "{self.excerpt}"')
        if self.url_or_path:
            lines.append(f"Location   : {self.url_or_path}")
        lines.append(f"Confidence : {self.confidence:.0%}")
        return "\n".join(lines)


@dataclass
class AttributedStatement:
    """
    A single factual statement paired with its attributions.
    """
    statement: str
    attributions: list[Attribution]
    is_synthesised: bool = False   # True if it came from MemoryFusion
    uncertainty_flag: bool = False

    def inline(self) -> str:
        """Statement followed by inline citations."""
        citations = " ".join(a.to_inline_citation() for a in self.attributions)
        return f"{self.statement} {citations}"


@dataclass
class AttributedAnswer:
    """
    A complete, fully-attributed answer ready for the CEO.
    """
    query: str
    statements: list[AttributedStatement]
    all_attributions: list[Attribution]
    generated_at: str = ""
    coverage_score: float = 0.0   # fraction of statements that have ≥1 citation

    def to_readable(self, show_full_citations: bool = False) -> str:
        """Render the answer in a human-readable, auditable format."""
        lines = [
            f"Query: {self.query}",
            f"Generated: {self.generated_at}",
            f"Citation coverage: {self.coverage_score:.0%}",
            "",
            "─" * 60,
            "ANSWER",
            "─" * 60,
        ]
        for s in self.statements:
            lines.append(s.inline())

        lines += ["", "─" * 60, "SOURCES", "─" * 60]
        seen: set[str] = set()
        for attr in self.all_attributions:
            if attr.attribution_id in seen:
                continue
            seen.add(attr.attribution_id)
            lines.append("")
            lines.append(attr.to_full_citation())

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Participant extractor
# ---------------------------------------------------------------------------

def _extract_participants(metadata: dict, text: str) -> list[str]:
    """
    Extract person names from metadata and text.
    In production: use NER / a contacts lookup.
    """
    participants: list[str] = []

    # From metadata
    for key in ("person", "sender", "recipients", "participants", "author"):
        val = metadata.get(key)
        if isinstance(val, str) and val:
            participants.append(val)
        elif isinstance(val, list):
            participants.extend(val)

    # From text (capitalised two-word names)
    for m in re.finditer(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text):
        name = m.group(1)
        if name not in participants:
            participants.append(name)

    return list(dict.fromkeys(participants))[:5]   # dedupe, cap at 5


# ---------------------------------------------------------------------------
# Attribution builder
# ---------------------------------------------------------------------------

class AttributionBuilder:
    """
    Builds Attribution objects from raw MemoryChunks or FusedMemory objects.
    """

    def from_chunk(self, chunk: MemoryChunk, confidence: float = 1.0) -> Attribution:
        date_str = (
            datetime.fromtimestamp(chunk.timestamp).strftime("%b %d, %Y")
            if chunk.timestamp else "Unknown date"
        )
        participants = _extract_participants(chunk.metadata, chunk.text)
        excerpt = chunk.text[:150].rstrip() + ("…" if len(chunk.text) > 150 else "")
        attr_id = hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:12]

        return Attribution(
            attribution_id=attr_id,
            source_type=chunk.source_type,
            source_date=date_str,
            source_date_ts=chunk.timestamp or 0.0,
            participants=participants,
            chunk_id=chunk.chunk_id,
            excerpt=excerpt,
            confidence=confidence,
            url_or_path=chunk.metadata.get("url", chunk.metadata.get("path", "")),
        )

    def from_fused_memory(self, fm: FusedMemory) -> list[Attribution]:
        return [self.from_chunk(c, confidence=fm.confidence) for c in fm.source_chunks]

    def from_result(self, result: HybridSearchResult) -> Attribution:
        return self.from_chunk(result.chunk, confidence=result.final_score)


# ---------------------------------------------------------------------------
# Statement → attribution linker
# ---------------------------------------------------------------------------

class StatementAttributor:
    """
    Given a list of natural-language statements (sentences in an answer)
    and a list of available attributions, links each statement to the
    most relevant attributions.

    Strategy: keyword overlap between statement and chunk excerpts.
    """

    def link(
        self,
        statements: list[str],
        attributions: list[Attribution],
        max_citations_per_statement: int = 3,
    ) -> list[AttributedStatement]:
        attributed = []
        for stmt in statements:
            stmt_tokens = set(re.findall(r"\b[a-z0-9]+\b", stmt.lower()))
            scored = []
            for attr in attributions:
                excerpt_tokens = set(re.findall(r"\b[a-z0-9]+\b", attr.excerpt.lower()))
                overlap = len(stmt_tokens & excerpt_tokens)
                if overlap > 0:
                    scored.append((overlap, attr))
            scored.sort(key=lambda x: x[0], reverse=True)
            best = [a for _, a in scored[:max_citations_per_statement]]
            attributed.append(AttributedStatement(
                statement=stmt,
                attributions=best,
                uncertainty_flag=(not best),
            ))
        return attributed


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class SourceAttributionEngine:
    """
    Full pipeline:
      1. Build attributions from raw chunks / fused memories
      2. Link attributions to each statement in the answer
      3. Return a fully attributed, auditable answer

    Usage
    -----
    engine = SourceAttributionEngine()
    attributed = engine.attribute(
        query="What is happening with Schneider?",
        answer_text="Approval is delayed. Timeline shifted 2 weeks.",
        results=hybrid_results,
    )
    print(attributed.to_readable())
    """

    def __init__(self) -> None:
        self._builder = AttributionBuilder()
        self._attributor = StatementAttributor()

    def attribute(
        self,
        query: str,
        answer_text: str,
        results: Optional[list[HybridSearchResult]] = None,
        fused_memories: Optional[list[FusedMemory]] = None,
    ) -> AttributedAnswer:

        # 1. Build attribution pool
        all_attributions: list[Attribution] = []

        if results:
            for r in results:
                all_attributions.append(self._builder.from_result(r))

        if fused_memories:
            for fm in fused_memories:
                all_attributions.extend(self._builder.from_fused_memory(fm))

        # Deduplicate by chunk_id
        seen_chunks: set[str] = set()
        unique_attributions: list[Attribution] = []
        for a in all_attributions:
            if a.chunk_id not in seen_chunks:
                seen_chunks.add(a.chunk_id)
                unique_attributions.append(a)

        # 2. Split answer into statements
        statements = self._split_into_statements(answer_text)

        # 3. Link statements to attributions
        attributed_statements = self._attributor.link(statements, unique_attributions)

        # 4. Coverage score
        cited = sum(1 for s in attributed_statements if s.attributions)
        coverage = cited / len(attributed_statements) if attributed_statements else 0.0

        return AttributedAnswer(
            query=query,
            statements=attributed_statements,
            all_attributions=unique_attributions,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            coverage_score=coverage,
        )

    def attribute_from_chunks(
        self,
        query: str,
        answer_text: str,
        chunks: list[MemoryChunk],
    ) -> AttributedAnswer:
        """Convenience wrapper when you have raw chunks, not HybridSearchResults."""
        attributions = [self._builder.from_chunk(c) for c in chunks]

        statements = self._split_into_statements(answer_text)
        attributed_statements = self._attributor.link(statements, attributions)

        cited = sum(1 for s in attributed_statements if s.attributions)
        coverage = cited / len(attributed_statements) if attributed_statements else 0.0

        return AttributedAnswer(
            query=query,
            statements=attributed_statements,
            all_attributions=attributions,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            coverage_score=coverage,
        )

    # ------------------------------------------------------------------

    def _split_into_statements(self, text: str) -> list[str]:
        """Split answer text into individual factual statements."""
        # Split on sentence boundaries
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchEngine

    now = _time.time()

    chunks = [
        MemoryChunk("email_001", "email",
                    "Approval delayed. Waiting for regulator sign-off on Smart Meter deployment.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter",
                              "person": "John Smith"},
                    timestamp=now - 86400 * 2),
        MemoryChunk("meeting_002", "meeting",
                    "KEI confirmed Smart Meter timeline shifted by 2 weeks due to approval hold.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    timestamp=now - 86400 * 5),
        MemoryChunk("risk_003", "risk",
                    "Risk identified: Q2 delivery at risk if regulator response delayed.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    timestamp=now - 86400 * 1),
    ]

    query = "What is happening with Schneider Electric?"
    answer = (
        "Approval is delayed pending regulator response. "
        "The Smart Meter timeline has shifted by two weeks. "
        "Q2 delivery is at risk if the approval is not received soon."
    )

    engine = SourceAttributionEngine()
    attributed = engine.attribute_from_chunks(query, answer, chunks)

    print("=" * 70)
    print("PHASE 4 - TASK 6: SOURCE ATTRIBUTION ENGINE — DEMO")
    print("=" * 70)
    print(attributed.to_readable(show_full_citations=True))
