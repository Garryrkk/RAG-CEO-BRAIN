
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from task4_context_assembly.context_assembly import AssembledContext, ContextSection
from task7_ranking.ranking_system import ScoredResult


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class InclusionDecision(str, Enum):
    INCLUDE_FULL    = "include_full"      # fits; include verbatim
    INCLUDE_SUMMARY = "include_summary"   # truncated / summarised
    EXCLUDE         = "exclude"           # below budget or below quality threshold


CHARS_PER_TOKEN = 4           # rough estimate
DEFAULT_BUDGET_TOKENS = 6000  # conservative, leaves room for system prompt + answer
MAX_SECTION_TOKENS = 1500
MIN_QUALITY_SCORE = 0.20      # chunks below this are excluded regardless


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BudgetAllocation:
    """How token budget is split across sections."""
    section_name: str
    allocated_tokens: int
    priority: int       # 0 = highest


@dataclass
class ContextWindowDecision:
    """Decision record for one chunk / section."""
    item_id: str
    decision: InclusionDecision
    original_tokens: int
    final_tokens: int
    reason: str


@dataclass
class ManagedContext:
    """
    The final, token-managed context object passed to the LLM.
    """
    query: str
    prompt_block: str             # ready for LLM
    total_tokens_used: int
    token_budget: int
    inclusion_decisions: list[ContextWindowDecision]
    sections_included: list[str]
    sections_excluded: list[str]
    compression_ratio: float      # original_tokens / final_tokens

    def budget_utilisation(self) -> str:
        pct = 100 * self.total_tokens_used / max(self.token_budget, 1)
        return f"{self.total_tokens_used}/{self.token_budget} tokens ({pct:.1f}%)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _summarise(text: str, target_tokens: int) -> str:
    """
    Trim text to target_tokens while preserving sentence integrity.
    In production: replace with an LLM summarisation call.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return text
    truncated = text[:target_chars]
    # Snap to last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > target_chars // 2:
        return truncated[:last_period + 1] + " [summarised]"
    return truncated + "… [summarised]"


def _section_priority(section_title: str) -> int:
    """
    Assign a display / budget priority to a context section.
    Lower number = higher priority.
    """
    order = {
        "company overview":          0,
        "identity":                  0,
        "active risks":              1,
        "risks":                     1,
        "risks & concerns":          1,
        "open commitments":          2,
        "overdue / escalated commitments": 2,
        "recent communications":     3,
        "recent activity":           3,
        "timeline":                  4,
        "active projects":           4,
        "associated projects":       4,
        "at-risk projects":          4,
        "open items":                5,
        "stakeholders":              5,
        "source conversations":      6,
        "deadlines":                 6,
        "escalations":               6,
        "relevant memory":           7,
    }
    return order.get(section_title.lower(), 8)


# ---------------------------------------------------------------------------
# Budget allocator
# ---------------------------------------------------------------------------

class BudgetAllocator:
    """
    Divides a total token budget across context sections based on their priority.
    Higher-priority sections get more tokens.
    """

    def allocate(
        self,
        sections: list[ContextSection],
        total_budget: int,
    ) -> list[BudgetAllocation]:
        if not sections:
            return []

        # Score each section by inverse priority (lower priority number = higher weight)
        max_pri = max(_section_priority(s.title) for s in sections) + 1
        weights = [max_pri - _section_priority(s.title) for s in sections]
        total_weight = sum(weights) or 1

        allocations = []
        remaining = total_budget
        for i, section in enumerate(sections):
            share = int(total_budget * weights[i] / total_weight)
            share = max(share, 100)   # floor: every section gets at least 100 tokens
            share = min(share, remaining, MAX_SECTION_TOKENS)
            remaining -= share
            allocations.append(BudgetAllocation(
                section_name=section.title,
                allocated_tokens=share,
                priority=_section_priority(section.title),
            ))

        return allocations


# ---------------------------------------------------------------------------
# Main context window manager
# ---------------------------------------------------------------------------

class ContextWindowManager:
    """
    Takes an AssembledContext (from Task 4) and produces a ManagedContext
    that fits within the LLM token budget.

    Rules:
      1. Critical sections (risks, open commitments) always enter first.
      2. Each section is allocated a proportional token budget.
      3. Content exceeding the section budget is summarised.
      4. Sections that would contribute < MIN_QUALITY_SCORE are excluded.
      5. Source references are always preserved.

    Usage
    -----
    manager = ContextWindowManager(budget_tokens=6000)
    managed = manager.compress(assembled_context)
    llm_input = managed.prompt_block
    """

    def __init__(
        self,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        min_quality: float = MIN_QUALITY_SCORE,
    ) -> None:
        self._budget = budget_tokens
        self._min_quality = min_quality
        self._allocator = BudgetAllocator()

    def compress(self, context: AssembledContext) -> ManagedContext:
        sections = sorted(context.sections, key=lambda s: _section_priority(s.title))
        allocations = self._allocator.allocate(sections, self._budget)
        alloc_map = {a.section_name: a for a in allocations}

        decisions: list[ContextWindowDecision] = []
        included_sections: list[str] = []
        excluded_sections: list[str] = []
        final_blocks: list[str] = []
        total_tokens = 0

        # Header block (cheap – always include)
        header = (
            f"CONTEXT | Query: {context.query} | "
            f"Strategy: {context.strategy} | "
            f"Sources: {context.source_count} | "
            f"Assembled: {context.assembled_at}\n"
        )
        total_tokens += _count_tokens(header)
        final_blocks.append(header)

        for section in sections:
            alloc = alloc_map.get(section.title)
            budget = alloc.allocated_tokens if alloc else 200
            original_tokens = _count_tokens(section.content)

            if total_tokens + min(original_tokens, budget) > self._budget:
                decisions.append(ContextWindowDecision(
                    item_id=section.title,
                    decision=InclusionDecision.EXCLUDE,
                    original_tokens=original_tokens,
                    final_tokens=0,
                    reason="Token budget exhausted",
                ))
                excluded_sections.append(section.title)
                continue

            if original_tokens <= budget:
                # Fits fully
                content = section.content
                dec = InclusionDecision.INCLUDE_FULL
                reason = "Fits within section budget"
            else:
                # Summarise
                content = _summarise(section.content, budget)
                dec = InclusionDecision.INCLUDE_SUMMARY
                reason = f"Summarised from {original_tokens} to {budget} tokens"

            final_tokens = _count_tokens(content)
            total_tokens += final_tokens
            decisions.append(ContextWindowDecision(
                item_id=section.title,
                decision=dec,
                original_tokens=original_tokens,
                final_tokens=final_tokens,
                reason=reason,
            ))
            included_sections.append(section.title)
            final_blocks.append(f"=== {section.title.upper()} ===\n{content}\n")

        prompt_block = "\n".join(final_blocks)
        original_total = _count_tokens(context.to_prompt_block())
        compression = round(original_total / max(total_tokens, 1), 2)

        return ManagedContext(
            query=context.query,
            prompt_block=prompt_block,
            total_tokens_used=total_tokens,
            token_budget=self._budget,
            inclusion_decisions=decisions,
            sections_included=included_sections,
            sections_excluded=excluded_sections,
            compression_ratio=compression,
        )

    def compress_from_ranked(
        self,
        query: str,
        ranked_results: list[ScoredResult],
        strategy: str = "general_summary",
    ) -> ManagedContext:
        """
        Convenience method: takes ranked results, assembles context on-the-fly,
        then compresses it.
        """
        from task4_context_assembly.context_assembly import (
            ContextAssemblyEngine,
        )
        from task3_hybrid_search.hybrid_search import HybridSearchResult

        hybrid_results = [sr.result for sr in ranked_results]
        assembled = ContextAssemblyEngine().assemble(query, hybrid_results, strategy)
        return self.compress(assembled)


# ---------------------------------------------------------------------------
# Adaptive budget profiles
# ---------------------------------------------------------------------------

class AdaptiveBudgetManager(ContextWindowManager):
    """
    Adjusts the budget dynamically based on:
    - Number of sections available
    - Query complexity (longer queries need more space)
    - Reserved space for the answer
    """

    ANSWER_RESERVE_TOKENS = 800

    def __init__(
        self,
        model_context_limit: int = 8000,
        system_prompt_tokens: int = 500,
    ) -> None:
        available = model_context_limit - system_prompt_tokens - self.ANSWER_RESERVE_TOKENS
        super().__init__(budget_tokens=max(available, 2000))

    def compress(self, context: AssembledContext) -> ManagedContext:
        # Adjust budget based on how many sections we have
        n_sections = len(context.sections)
        # More sections → more budget needed (up to our cap)
        adjusted = min(self._budget, n_sections * 900)
        self._budget = max(adjusted, 2000)
        return super().compress(context)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchEngine
    from task4_context_assembly.context_assembly import ContextAssemblyEngine

    now = _t = _time.time()

    corpus = [
        MemoryChunk("email_001", "email",
                    "Approval delayed. Waiting for regulator sign-off on Smart Meter.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    timestamp=now - 86400 * 2),
        MemoryChunk("meeting_002", "meeting",
                    "KEI confirmed Smart Meter timeline shifted by 2 weeks.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    timestamp=now - 86400 * 5),
        MemoryChunk("risk_003", "risk",
                    "Risk: Q2 delivery at risk if regulator response delayed.",
                    metadata={"company": "Schneider Electric"},
                    timestamp=now - 86400 * 1),
        MemoryChunk("commitment_004", "commitment",
                    "John Smith to follow up with regulator by March 25.",
                    metadata={"person": "John Smith", "status": "open"},
                    timestamp=now - 86400 * 3),
    ]

    query = "What is happening with Schneider Electric?"
    search_engine = HybridSearchEngine()
    results = search_engine.search(query, corpus, filters={"company": "Schneider Electric"})
    assembled = ContextAssemblyEngine().assemble(query, results, strategy="company_360")

    manager = ContextWindowManager(budget_tokens=2000)   # tight budget for demo
    managed = manager.compress(assembled)

    print("=" * 70)
    print("PHASE 4 - TASK 8: CONTEXT WINDOW MANAGEMENT — DEMO")
    print("=" * 70)
    print(f"\nBudget utilisation : {managed.budget_utilisation()}")
    print(f"Compression ratio  : {managed.compression_ratio}x")
    print(f"Sections included  : {managed.sections_included}")
    print(f"Sections excluded  : {managed.sections_excluded}")
    print("\nDecisions:")
    for d in managed.inclusion_decisions:
        print(f"  [{d.decision.value:18}] {d.item_id:<30} "
              f"{d.original_tokens:>5} → {d.final_tokens:>5} tokens  | {d.reason}")
    print("\n--- FINAL PROMPT BLOCK ---")
    print(managed.prompt_block[:1000], "…" if len(managed.prompt_block) > 1000 else "")
