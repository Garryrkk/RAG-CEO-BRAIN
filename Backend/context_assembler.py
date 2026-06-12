
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from task3_hybrid_search.hybrid_search import HybridSearchResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContextSection:
    """A labelled section of assembled context."""
    title: str
    content: str
    source_ids: list[str] = field(default_factory=list)
    priority: int = 0   # lower = appears first


@dataclass
class AssembledContext:
    """
    The complete, structured context handed to the answer generation layer.
    """
    query: str
    strategy: str
    sections: list[ContextSection]
    total_tokens_estimate: int = 0
    source_count: int = 0
    assembled_at: str = ""

    def to_prompt_block(self) -> str:
        """Render all sections into a single LLM-ready text block."""
        lines = [f"=== ORGANISATIONAL MEMORY CONTEXT ===",
                 f"Query: {self.query}",
                 f"Strategy: {self.strategy}",
                 f"Sources: {self.source_count}",
                 ""]
        for section in sorted(self.sections, key=lambda s: s.priority):
            lines.append(f"--- {section.title.upper()} ---")
            lines.append(section.content.strip())
            lines.append("")
        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base assembler
# ---------------------------------------------------------------------------

class BaseContextAssembler:
    """
    Base class for all context assemblers.
    Subclasses override `assemble()` for strategy-specific logic.
    """

    MAX_SECTION_CHARS = 1200   # per section
    CHARS_PER_TOKEN   = 4      # rough estimate

    def assemble(
        self,
        query: str,
        results: list[HybridSearchResult],
    ) -> AssembledContext:
        raise NotImplementedError

    # Shared utilities ---------------------------------------------------

    def _snippet(self, text: str, max_chars: int = 400) -> str:
        """Return a clean snippet, truncating gracefully at sentence boundary."""
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        last_period = cut.rfind(".")
        if last_period > max_chars // 2:
            return cut[:last_period + 1] + " …"
        return cut + " …"

    def _results_by_type(
        self, results: list[HybridSearchResult]
    ) -> dict[str, list[HybridSearchResult]]:
        grouped: dict[str, list[HybridSearchResult]] = {}
        for r in results:
            st = r.chunk.source_type
            grouped.setdefault(st, []).append(r)
        return grouped

    def _format_timestamp(self, ts: float) -> str:
        if not ts:
            return "unknown date"
        return datetime.fromtimestamp(ts).strftime("%b %d, %Y")

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN

    def _build_context(
        self,
        query: str,
        strategy: str,
        sections: list[ContextSection],
        source_count: int,
    ) -> AssembledContext:
        full_text = " ".join(s.content for s in sections)
        return AssembledContext(
            query=query,
            strategy=strategy,
            sections=sections,
            total_tokens_estimate=self._estimate_tokens(full_text),
            source_count=source_count,
            assembled_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )


# ---------------------------------------------------------------------------
# Strategy-specific assemblers
# ---------------------------------------------------------------------------

class PersonProfileAssembler(BaseContextAssembler):
    """
    Assembles context for person queries.
    Sections: Identity → Recent Activity → Commitments → Projects
    """

    def assemble(self, query: str, results: list[HybridSearchResult]) -> AssembledContext:
        grouped = self._results_by_type(results)
        sections: list[ContextSection] = []

        # Identity
        entities = grouped.get("entity", grouped.get("person_entity", []))
        if entities:
            content = "\n".join(
                f"• {self._snippet(r.chunk.text, 300)}"
                for r in entities[:2]
            )
            sections.append(ContextSection(
                title="Identity",
                content=content,
                source_ids=[r.chunk.chunk_id for r in entities[:2]],
                priority=0,
            ))

        # Recent Activity (emails + meetings + conversations)
        activity = (
            grouped.get("email", []) +
            grouped.get("meeting", []) +
            grouped.get("conversation", [])
        )
        activity.sort(key=lambda r: r.chunk.timestamp, reverse=True)
        if activity:
            lines = []
            for r in activity[:5]:
                date = self._format_timestamp(r.chunk.timestamp)
                lines.append(f"[{date}] {r.chunk.source_type.upper()}: "
                              f"{self._snippet(r.chunk.text, 200)}")
            sections.append(ContextSection(
                title="Recent Activity",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in activity[:5]],
                priority=1,
            ))

        # Commitments
        commitments = grouped.get("commitment", [])
        if commitments:
            lines = [
                f"• [{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 200)}"
                for r in commitments[:5]
            ]
            sections.append(ContextSection(
                title="Commitments",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in commitments[:5]],
                priority=2,
            ))

        # Projects
        projects = grouped.get("project", [])
        if projects:
            lines = [f"• {self._snippet(r.chunk.text, 150)}" for r in projects[:3]]
            sections.append(ContextSection(
                title="Associated Projects",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in projects[:3]],
                priority=3,
            ))

        return self._build_context(query, "person_profile", sections, len(results))


class Company360Assembler(BaseContextAssembler):
    """
    Full 360-degree view of a company.
    Sections: Overview → Current Status → Recent Communications →
              Open Commitments → Risks → Evidence
    """

    def assemble(self, query: str, results: list[HybridSearchResult]) -> AssembledContext:
        grouped = self._results_by_type(results)
        sections: list[ContextSection] = []

        # Overview
        entity = grouped.get("entity", grouped.get("company_memory", []))
        if entity:
            sections.append(ContextSection(
                title="Company Overview",
                content=self._snippet(entity[0].chunk.text, 500),
                source_ids=[entity[0].chunk.chunk_id],
                priority=0,
            ))

        # Recent Communications
        comms = (
            grouped.get("email", []) +
            grouped.get("meeting", []) +
            grouped.get("conversation", [])
        )
        comms.sort(key=lambda r: r.chunk.timestamp, reverse=True)
        if comms:
            lines = []
            for r in comms[:6]:
                date = self._format_timestamp(r.chunk.timestamp)
                lines.append(f"[{date}] {r.chunk.source_type.upper()}: "
                              f"{self._snippet(r.chunk.text, 250)}")
            sections.append(ContextSection(
                title="Recent Communications",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in comms[:6]],
                priority=1,
            ))

        # Projects
        projects = grouped.get("project", [])
        if projects:
            lines = [f"• {self._snippet(r.chunk.text, 200)}" for r in projects[:4]]
            sections.append(ContextSection(
                title="Active Projects",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in projects[:4]],
                priority=2,
            ))

        # Open Commitments
        commitments = grouped.get("commitment", [])
        if commitments:
            lines = [
                f"• [{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 200)}"
                for r in commitments[:6]
            ]
            sections.append(ContextSection(
                title="Open Commitments",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in commitments[:6]],
                priority=3,
            ))

        # Risks
        risks = grouped.get("risk", [])
        if risks:
            lines = [
                f"⚠ {self._snippet(r.chunk.text, 200)}"
                for r in risks[:5]
            ]
            sections.append(ContextSection(
                title="Risks & Concerns",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in risks[:5]],
                priority=4,
            ))

        return self._build_context(query, "company_360", sections, len(results))


class ProjectStatusAssembler(BaseContextAssembler):
    """
    Project-centric view.
    Sections: Project Record → Timeline → Stakeholders →
              Recent Activity → Risks → Open Items
    """

    def assemble(self, query: str, results: list[HybridSearchResult]) -> AssembledContext:
        grouped = self._results_by_type(results)
        sections: list[ContextSection] = []

        # Project record
        proj = grouped.get("project", [])
        if proj:
            sections.append(ContextSection(
                title="Project Overview",
                content=self._snippet(proj[0].chunk.text, 500),
                source_ids=[proj[0].chunk.chunk_id],
                priority=0,
            ))

        # Timeline (sorted chronologically)
        timeline = grouped.get("timeline", [])
        timeline.sort(key=lambda r: r.chunk.timestamp)
        if timeline:
            lines = [
                f"[{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 180)}"
                for r in timeline[:8]
            ]
            sections.append(ContextSection(
                title="Timeline",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in timeline[:8]],
                priority=1,
            ))

        # Stakeholders
        stakeholders = grouped.get("stakeholder", [])
        if stakeholders:
            lines = [f"• {self._snippet(r.chunk.text, 150)}" for r in stakeholders[:5]]
            sections.append(ContextSection(
                title="Stakeholders",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in stakeholders[:5]],
                priority=2,
            ))

        # Risks
        risks = grouped.get("risk", [])
        if risks:
            lines = [f"⚠ {self._snippet(r.chunk.text, 200)}" for r in risks[:4]]
            sections.append(ContextSection(
                title="Risks",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in risks[:4]],
                priority=3,
            ))

        # Open Commitments
        commitments = grouped.get("commitment", [])
        if commitments:
            lines = [
                f"• {self._snippet(r.chunk.text, 200)}"
                for r in commitments[:5]
            ]
            sections.append(ContextSection(
                title="Open Items",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in commitments[:5]],
                priority=4,
            ))

        return self._build_context(query, "project_status", sections, len(results))


class RiskRegisterAssembler(BaseContextAssembler):
    """
    Risk-focused view.
    Sections: Active Risks → At-Risk Projects → Overdue Commitments → Escalations
    """

    def assemble(self, query: str, results: list[HybridSearchResult]) -> AssembledContext:
        grouped = self._results_by_type(results)
        sections: list[ContextSection] = []

        risks = grouped.get("risk", [])
        risks.sort(key=lambda r: r.final_score, reverse=True)
        if risks:
            lines = [
                f"⚠ [{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 250)}"
                for r in risks[:8]
            ]
            sections.append(ContextSection(
                title="Active Risks",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in risks[:8]],
                priority=0,
            ))

        projects = grouped.get("project", [])
        if projects:
            lines = [f"• {self._snippet(r.chunk.text, 200)}" for r in projects[:4]]
            sections.append(ContextSection(
                title="At-Risk Projects",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in projects[:4]],
                priority=1,
            ))

        commitments = grouped.get("commitment", [])
        if commitments:
            lines = [
                f"• {self._snippet(r.chunk.text, 200)}"
                for r in commitments[:5]
            ]
            sections.append(ContextSection(
                title="Overdue / Escalated Commitments",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in commitments[:5]],
                priority=2,
            ))

        escalations = [
            r for r in grouped.get("email", [])
            if "escalat" in (r.chunk.metadata.get("tags") or [])
        ]
        if escalations:
            lines = [
                f"[{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 200)}"
                for r in escalations[:4]
            ]
            sections.append(ContextSection(
                title="Escalations",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in escalations[:4]],
                priority=3,
            ))

        return self._build_context(query, "risk_register", sections, len(results))


class CommitmentTrackerAssembler(BaseContextAssembler):
    """
    Commitment-centric view.
    Sections: Open Commitments → Source Conversations → Deadlines → Projects
    """

    def assemble(self, query: str, results: list[HybridSearchResult]) -> AssembledContext:
        grouped = self._results_by_type(results)
        sections: list[ContextSection] = []

        commitments = grouped.get("commitment", [])
        if commitments:
            lines = [
                f"• [{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 250)}"
                for r in commitments[:8]
            ]
            sections.append(ContextSection(
                title="Open Commitments",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in commitments[:8]],
                priority=0,
            ))

        conversations = grouped.get("conversation", grouped.get("meeting", []))
        if conversations:
            lines = [
                f"[{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 200)}"
                for r in conversations[:5]
            ]
            sections.append(ContextSection(
                title="Source Conversations",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in conversations[:5]],
                priority=1,
            ))

        timeline = grouped.get("timeline", [])
        deadlines = [r for r in timeline
                     if "deadline" in str(r.chunk.metadata.get("type", ""))]
        if deadlines:
            lines = [
                f"[{self._format_timestamp(r.chunk.timestamp)}] "
                f"{self._snippet(r.chunk.text, 150)}"
                for r in deadlines[:5]
            ]
            sections.append(ContextSection(
                title="Deadlines",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in deadlines[:5]],
                priority=2,
            ))

        return self._build_context(query, "commitment_tracker", sections, len(results))


class GeneralSummaryAssembler(BaseContextAssembler):
    """Fallback assembler for general queries."""

    def assemble(self, query: str, results: list[HybridSearchResult]) -> AssembledContext:
        grouped = self._results_by_type(results)
        sections: list[ContextSection] = []

        all_items = sorted(results, key=lambda r: r.chunk.timestamp, reverse=True)
        if all_items:
            lines = []
            for r in all_items[:8]:
                date = self._format_timestamp(r.chunk.timestamp)
                lines.append(f"[{date}] {r.chunk.source_type.upper()}: "
                              f"{self._snippet(r.chunk.text, 250)}")
            sections.append(ContextSection(
                title="Relevant Memory",
                content="\n".join(lines),
                source_ids=[r.chunk.chunk_id for r in all_items[:8]],
                priority=0,
            ))

        return self._build_context(query, "general_summary", sections, len(results))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ASSEMBLERS: dict[str, BaseContextAssembler] = {
    "person_profile":    PersonProfileAssembler(),
    "company_360":       Company360Assembler(),
    "project_status":    ProjectStatusAssembler(),
    "risk_register":     RiskRegisterAssembler(),
    "commitment_tracker": CommitmentTrackerAssembler(),
    "general_summary":   GeneralSummaryAssembler(),
}


class ContextAssemblyEngine:
    """
    Takes hybrid search results and a strategy name,
    returns a fully assembled AssembledContext.

    Usage
    -----
    engine = ContextAssemblyEngine()
    context = engine.assemble(
        query="What is happening with Schneider Electric?",
        results=hybrid_results,
        strategy="company_360",
    )
    print(context.to_prompt_block())
    """

    def assemble(
        self,
        query: str,
        results: list[HybridSearchResult],
        strategy: str = "general_summary",
    ) -> AssembledContext:
        assembler = _ASSEMBLERS.get(strategy, _ASSEMBLERS["general_summary"])
        return assembler.assemble(query, results)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchEngine

    now = _time.time()

    corpus = [
        MemoryChunk("email_001", "email",
                    "Smart Meter Deployment approval delayed pending regulator response.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    timestamp=now - 86400 * 2),
        MemoryChunk("meeting_002", "meeting",
                    "Discussed Smart Meter rollout timeline. Shifted by 2 weeks.",
                    metadata={"company": "KEI", "project": "Smart Meter"},
                    timestamp=now - 86400 * 5),
        MemoryChunk("commitment_003", "commitment",
                    "John Smith to follow up on regulator approval by March 25.",
                    metadata={"person": "John Smith", "status": "open"},
                    timestamp=now - 86400 * 3),
        MemoryChunk("risk_004", "risk",
                    "Risk: Q2 delivery at risk if approval not received by April 1.",
                    metadata={"company": "Schneider Electric"},
                    timestamp=now - 86400 * 1),
    ]

    search_engine = HybridSearchEngine()
    assembly_engine = ContextAssemblyEngine()

    query = "What is happening with Schneider Electric?"
    results = search_engine.search(
        query=query,
        corpus=corpus,
        filters={"company": "Schneider Electric"},
        top_k=10,
    )

    context = assembly_engine.assemble(query, results, strategy="company_360")

    print("=" * 70)
    print("PHASE 4 - TASK 4: CONTEXT ASSEMBLY ENGINE — DEMO")
    print("=" * 70)
    print(context.to_prompt_block())
    print(f"\nEstimated tokens: {context.total_tokens_estimate}")
    print(f"Sections: {[s.title for s in context.sections]}")
