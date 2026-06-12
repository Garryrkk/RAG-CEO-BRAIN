
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Re-use enums / types from Task 1
from task1_query_understanding.query_understanding import (
    QueryCategory,
    QueryIntent,
    TemporalScope,
    ClassifiedQuery,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemorySourceType(str, Enum):
    ENTITY = "entity"
    EMAIL = "email"
    MEETING = "meeting"
    DOCUMENT = "document"
    COMMITMENT = "commitment"
    RISK = "risk"
    PROJECT = "project"
    CONVERSATION = "conversation"
    TIMELINE = "timeline"
    STAKEHOLDER = "stakeholder"


class RetrievalPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemorySourceSpec:
    """
    A single memory source that should be queried as part of a retrieval plan.
    """
    source_type: MemorySourceType
    priority: RetrievalPriority
    max_results: int              # how many records to fetch from this source
    recency_weight: float         # 0-1, how much to favour recent records
    filters: dict = field(default_factory=dict)   # e.g. {"status": "open"}
    description: str = ""


@dataclass
class RetrievalPlan:
    """
    A complete, intent-aware retrieval plan for a classified query.
    """
    query: ClassifiedQuery
    sources: list[MemorySourceSpec]
    search_modes: list[str]       # e.g. ["semantic", "keyword", "metadata"]
    max_total_results: int
    assembly_strategy: str        # how to assemble context from results
    ranking_hints: list[str]      # signals to boost in Task 7
    notes: str = ""

    def ordered_sources(self) -> list[MemorySourceSpec]:
        """Return sources ordered by priority."""
        priority_order = {
            RetrievalPriority.CRITICAL: 0,
            RetrievalPriority.HIGH: 1,
            RetrievalPriority.MEDIUM: 2,
            RetrievalPriority.LOW: 3,
        }
        return sorted(self.sources, key=lambda s: priority_order[s.priority])


# ---------------------------------------------------------------------------
# Plan builders – one per category
# ---------------------------------------------------------------------------

class PersonQueryPlanBuilder:
    """
    Builds retrieval plans for queries about a specific person.

    Retrieves: Person Entity, Conversations, Commitments, Projects, Emails.
    """

    def build(self, query: ClassifiedQuery) -> RetrievalPlan:
        entity_name = query.primary_entity().text if query.primary_entity() else ""

        sources = [
            MemorySourceSpec(
                source_type=MemorySourceType.ENTITY,
                priority=RetrievalPriority.CRITICAL,
                max_results=1,
                recency_weight=0.2,
                filters={"name": entity_name},
                description="Core person entity record",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.CONVERSATION,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.8,
                filters={"participant": entity_name},
                description="Recent conversations involving this person",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.COMMITMENT,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.5,
                filters={"owner": entity_name, "status": "open"},
                description="Open commitments owned by or involving this person",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.PROJECT,
                priority=RetrievalPriority.MEDIUM,
                max_results=5,
                recency_weight=0.4,
                filters={"stakeholder": entity_name},
                description="Projects this person is linked to",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.EMAIL,
                priority=RetrievalPriority.MEDIUM,
                max_results=8,
                recency_weight=0.9,
                filters={"participant": entity_name},
                description="Recent emails to/from this person",
            ),
        ]

        return RetrievalPlan(
            query=query,
            sources=sources,
            search_modes=["semantic", "keyword", "relationship"],
            max_total_results=30,
            assembly_strategy="person_profile",
            ranking_hints=["recency", "relationship_strength"],
            notes=f"Person query for '{entity_name}'. "
                  f"Prioritises entity record first, then communications.",
        )


class CompanyQueryPlanBuilder:
    """
    Builds retrieval plans for queries about a company / account.

    Retrieves: Company Memory, Projects, Conversations, Commitments,
               Risks, Emails, Meetings.
    """

    def build(self, query: ClassifiedQuery) -> RetrievalPlan:
        entity_name = query.primary_entity().text if query.primary_entity() else ""

        sources = [
            MemorySourceSpec(
                source_type=MemorySourceType.ENTITY,
                priority=RetrievalPriority.CRITICAL,
                max_results=1,
                recency_weight=0.1,
                filters={"name": entity_name},
                description="Core company entity record",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.PROJECT,
                priority=RetrievalPriority.HIGH,
                max_results=5,
                recency_weight=0.5,
                filters={"company": entity_name},
                description="Active projects linked to this company",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.CONVERSATION,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.9,
                filters={"company": entity_name},
                description="Recent conversations about this company",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.COMMITMENT,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.6,
                filters={"company": entity_name, "status": "open"},
                description="Open commitments for this company",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.RISK,
                priority=RetrievalPriority.HIGH,
                max_results=5,
                recency_weight=0.7,
                filters={"company": entity_name},
                description="Active risks associated with this company",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.EMAIL,
                priority=RetrievalPriority.MEDIUM,
                max_results=8,
                recency_weight=0.9,
                filters={"company": entity_name},
                description="Recent emails from/about this company",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.MEETING,
                priority=RetrievalPriority.MEDIUM,
                max_results=5,
                recency_weight=0.8,
                filters={"company": entity_name},
                description="Recent meeting notes about this company",
            ),
        ]

        return RetrievalPlan(
            query=query,
            sources=sources,
            search_modes=["semantic", "keyword", "metadata", "relationship", "timeline"],
            max_total_results=40,
            assembly_strategy="company_360",
            ranking_hints=["recency", "importance", "authority"],
            notes=f"Company query for '{entity_name}'. "
                  f"Full 360-degree memory retrieval.",
        )


class ProjectQueryPlanBuilder:
    """
    Builds retrieval plans for queries about a specific project.
    """

    def build(self, query: ClassifiedQuery) -> RetrievalPlan:
        entity_name = query.primary_entity().text if query.primary_entity() else ""

        sources = [
            MemorySourceSpec(
                source_type=MemorySourceType.PROJECT,
                priority=RetrievalPriority.CRITICAL,
                max_results=1,
                recency_weight=0.3,
                filters={"name": entity_name},
                description="Core project record",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.TIMELINE,
                priority=RetrievalPriority.HIGH,
                max_results=20,
                recency_weight=0.4,
                filters={"project": entity_name},
                description="Full project timeline and milestones",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.STAKEHOLDER,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.2,
                filters={"project": entity_name},
                description="Stakeholders involved in this project",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.DOCUMENT,
                priority=RetrievalPriority.MEDIUM,
                max_results=10,
                recency_weight=0.5,
                filters={"project": entity_name},
                description="Documents and deliverables for this project",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.RISK,
                priority=RetrievalPriority.HIGH,
                max_results=8,
                recency_weight=0.8,
                filters={"project": entity_name},
                description="Risks and blockers for this project",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.COMMITMENT,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.7,
                filters={"project": entity_name, "status": "open"},
                description="Open commitments within this project",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.MEETING,
                priority=RetrievalPriority.MEDIUM,
                max_results=5,
                recency_weight=0.9,
                filters={"project": entity_name},
                description="Recent meeting notes about this project",
            ),
        ]

        return RetrievalPlan(
            query=query,
            sources=sources,
            search_modes=["semantic", "keyword", "metadata", "timeline"],
            max_total_results=50,
            assembly_strategy="project_status",
            ranking_hints=["recency", "relevance", "milestone_proximity"],
            notes=f"Project query for '{entity_name}'. "
                  f"Timeline-forward retrieval strategy.",
        )


class RiskQueryPlanBuilder:
    """
    Builds retrieval plans for risk-related queries.
    """

    def build(self, query: ClassifiedQuery) -> RetrievalPlan:
        entity_name = query.primary_entity().text if query.primary_entity() else ""

        filters: dict = {}
        if entity_name:
            filters["related_entity"] = entity_name

        sources = [
            MemorySourceSpec(
                source_type=MemorySourceType.RISK,
                priority=RetrievalPriority.CRITICAL,
                max_results=20,
                recency_weight=0.8,
                filters={"status": "open", **filters},
                description="All open risk objects",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.PROJECT,
                priority=RetrievalPriority.HIGH,
                max_results=5,
                recency_weight=0.5,
                filters={"status": "at_risk", **filters},
                description="Projects flagged as at-risk",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.COMMITMENT,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.7,
                filters={"status": "overdue", **filters},
                description="Overdue or escalated commitments",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.CONVERSATION,
                priority=RetrievalPriority.MEDIUM,
                max_results=8,
                recency_weight=0.9,
                filters={"tags": "risk", **filters},
                description="Recent conversations mentioning risk",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.EMAIL,
                priority=RetrievalPriority.MEDIUM,
                max_results=5,
                recency_weight=0.9,
                filters={"tags": "escalation", **filters},
                description="Emails flagged as escalations",
            ),
        ]

        return RetrievalPlan(
            query=query,
            sources=sources,
            search_modes=["semantic", "keyword", "metadata"],
            max_total_results=40,
            assembly_strategy="risk_register",
            ranking_hints=["severity", "recency", "importance"],
            notes="Risk query. Prioritises open risk objects and overdue items.",
        )


class CommitmentQueryPlanBuilder:
    """
    Builds retrieval plans for commitment-related queries.
    """

    def build(self, query: ClassifiedQuery) -> RetrievalPlan:
        entity_name = query.primary_entity().text if query.primary_entity() else ""

        filters: dict = {}
        if entity_name:
            filters["entity"] = entity_name

        sources = [
            MemorySourceSpec(
                source_type=MemorySourceType.COMMITMENT,
                priority=RetrievalPriority.CRITICAL,
                max_results=20,
                recency_weight=0.5,
                filters={"status": "open", **filters},
                description="All open/unresolved commitments",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.CONVERSATION,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.8,
                filters={"tags": "commitment", **filters},
                description="Source conversations for these commitments",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.TIMELINE,
                priority=RetrievalPriority.MEDIUM,
                max_results=10,
                recency_weight=0.4,
                filters={"type": "deadline", **filters},
                description="Commitment deadlines on timeline",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.PROJECT,
                priority=RetrievalPriority.MEDIUM,
                max_results=5,
                recency_weight=0.3,
                filters={**filters},
                description="Projects associated with these commitments",
            ),
        ]

        return RetrievalPlan(
            query=query,
            sources=sources,
            search_modes=["semantic", "keyword", "metadata"],
            max_total_results=35,
            assembly_strategy="commitment_tracker",
            ranking_hints=["deadline_proximity", "recency", "importance"],
            notes="Commitment query. Surfaces open items and their deadlines.",
        )


class GeneralQueryPlanBuilder:
    """
    Fallback builder for unclassified or general queries.
    """

    def build(self, query: ClassifiedQuery) -> RetrievalPlan:
        sources = [
            MemorySourceSpec(
                source_type=MemorySourceType.CONVERSATION,
                priority=RetrievalPriority.HIGH,
                max_results=10,
                recency_weight=0.8,
                description="General recent conversations",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.DOCUMENT,
                priority=RetrievalPriority.MEDIUM,
                max_results=8,
                recency_weight=0.5,
                description="Relevant documents",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.EMAIL,
                priority=RetrievalPriority.MEDIUM,
                max_results=8,
                recency_weight=0.8,
                description="Recent emails",
            ),
            MemorySourceSpec(
                source_type=MemorySourceType.MEETING,
                priority=RetrievalPriority.MEDIUM,
                max_results=5,
                recency_weight=0.7,
                description="Recent meetings",
            ),
        ]

        return RetrievalPlan(
            query=query,
            sources=sources,
            search_modes=["semantic", "keyword"],
            max_total_results=25,
            assembly_strategy="general_summary",
            ranking_hints=["recency", "relevance"],
            notes="General query — broad retrieval across conversations, "
                  "documents, emails, and meetings.",
        )


# ---------------------------------------------------------------------------
# Main strategy engine
# ---------------------------------------------------------------------------

_BUILDERS: dict[QueryCategory, object] = {
    QueryCategory.PERSON:     PersonQueryPlanBuilder(),
    QueryCategory.COMPANY:    CompanyQueryPlanBuilder(),
    QueryCategory.PROJECT:    ProjectQueryPlanBuilder(),
    QueryCategory.RISK:       RiskQueryPlanBuilder(),
    QueryCategory.COMMITMENT: CommitmentQueryPlanBuilder(),
    QueryCategory.GENERAL:    GeneralQueryPlanBuilder(),
}


class RetrievalStrategyEngine:
    """
    Takes a ClassifiedQuery and returns a fully-formed RetrievalPlan.

    Usage
    -----
    engine = RetrievalStrategyEngine()
    plan = engine.build_plan(classified_query)
    for source in plan.ordered_sources():
        results = memory_store.fetch(source)
    """

    def build_plan(self, classified_query: ClassifiedQuery) -> RetrievalPlan:
        builder = _BUILDERS.get(
            classified_query.category, _BUILDERS[QueryCategory.GENERAL]
        )
        plan: RetrievalPlan = builder.build(classified_query)

        # Apply temporal adjustments
        plan = self._apply_temporal_adjustments(plan, classified_query.temporal_scope)

        return plan

    def _apply_temporal_adjustments(
        self, plan: RetrievalPlan, temporal: TemporalScope
    ) -> RetrievalPlan:
        """Adjust max_results and recency_weight based on temporal scope."""
        multiplier = {
            TemporalScope.RECENT:   0.5,   # fetch fewer, highly recent
            TemporalScope.MONTH:    0.75,
            TemporalScope.QUARTER:  1.0,
            TemporalScope.ALL_TIME: 1.5,   # broader sweep
            TemporalScope.SPECIFIC: 1.0,
        }.get(temporal, 1.0)

        for source in plan.sources:
            # Cap at sensible maximum
            source.max_results = min(int(source.max_results * multiplier), 30)
            if temporal == TemporalScope.RECENT:
                source.recency_weight = min(source.recency_weight + 0.2, 1.0)

        return plan


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from task1_query_understanding.query_understanding import QueryClassifier

    classifier = QueryClassifier()
    engine = RetrievalStrategyEngine()

    test_queries = [
        "What is happening with Schneider Electric?",
        "Who is John Smith?",
        "What commitments remain unresolved?",
        "What risks exist around the Smart Meter deployment?",
        "Summarize the KEI project status.",
    ]

    print("=" * 70)
    print("PHASE 4 - TASK 2: RETRIEVAL STRATEGY ENGINE — DEMO")
    print("=" * 70)

    for q in test_queries:
        classified = classifier.classify(q)
        plan = engine.build_plan(classified)

        print(f"\nQuery   : {q}")
        print(f"Category: {classified.category.value}  |  Intent: {classified.intent.value}")
        print(f"Plan    : assembly_strategy={plan.assembly_strategy}, "
              f"search_modes={plan.search_modes}")
        print(f"  Sources (ordered by priority):")
        for src in plan.ordered_sources():
            print(f"    [{src.priority.value.upper():8}] {src.source_type.value:15} "
                  f"max={src.max_results:3}  recency={src.recency_weight:.1f}  "
                  f"| {src.description}")
        print(f"  Total result budget: {plan.max_total_results}")
        print(f"  Ranking hints: {plan.ranking_hints}")
