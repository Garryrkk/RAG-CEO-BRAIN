

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchRequest

from app.memory.assembler import MemoryAssembler
from app.memory.memory_models import CompanyMemory, PersonMemory, ProjectMemory
from app.models.entities import (
    Commitment,
    CommitmentStatus,
    Company,
    Person,
    Project,
    Risk,
    RiskLevel,
    RiskStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Query Intent Classification
# ─────────────────────────────────────────────────────────────────────────────

class QueryIntent(str, enum.Enum):
    PERSON_QUERY = "person_query"         # "What is pending with John?"
    COMPANY_QUERY = "company_query"       # "What is happening with Schneider?"
    PROJECT_QUERY = "project_query"       # "Status of Smart Meter Rollout?"
    RISK_QUERY = "risk_query"             # "What risks exist?"
    COMMITMENT_QUERY = "commitment_query" # "What remains unresolved?"
    GENERAL_SEARCH = "general_search"     # "Find anything about infrastructure delay"
    SUMMARY_QUERY = "summary_query"       # "Summarize investor discussions"
    TIMELINE_QUERY = "timeline_query"     # "What happened last week?"
    UNKNOWN = "unknown"


@dataclass
class ResolvedEntity:
    """An entity resolved from the query."""
    entity_type: str                              # "person", "company", "project"
    entity_id: Optional[UUID]
    entity_name: str
    confidence: float = 1.0


@dataclass
class RetrievalPlan:
    """
    The plan for retrieving context for a given query.
    Built before any data fetching begins.
    """
    query: str
    intent: QueryIntent
    resolved_entities: list[ResolvedEntity] = field(default_factory=list)
    include_vector_search: bool = True
    vector_collection: str = "documents"
    vector_limit: int = 8
    include_person_memory: bool = False
    include_company_memory: bool = False
    include_project_memory: bool = False
    include_open_commitments: bool = False
    include_open_risks: bool = False
    include_timeline: bool = False
    date_filter_days: Optional[int] = None       # Only look at last N days
    priority_signals: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    """
    The assembled context — ready to be passed to the LLM.
    Everything the model needs, nothing it doesn't.
    """
    query: str
    intent: QueryIntent

    # Structured memory
    person_memories: list[PersonMemory] = field(default_factory=list)
    company_memories: list[CompanyMemory] = field(default_factory=list)
    project_memories: list[ProjectMemory] = field(default_factory=list)

    # Direct data
    open_commitments: list[dict] = field(default_factory=list)
    active_risks: list[dict] = field(default_factory=list)
    recent_interactions: list[dict] = field(default_factory=list)

    # Vector search results
    semantic_chunks: list[dict] = field(default_factory=list)

    # Metadata
    total_sources: int = 0
    retrieval_plan: Optional[RetrievalPlan] = None

    def to_llm_context(self) -> str:
        """
        Serialize all retrieved context into a string for LLM consumption.
        Ordered by relevance and importance.
        """
        sections = []

        if self.person_memories:
            sections.append("## PEOPLE CONTEXT")
            for pm in self.person_memories:
                ctx = pm.to_context_dict()
                sections.append(f"### {ctx['person']['name']}")
                sections.append(f"- Role: {ctx['person']['role']} at {', '.join(ctx['person']['companies'])}")
                if ctx["relationship"]["days_since_contact"] is not None:
                    sections.append(f"- Last contact: {ctx['relationship']['days_since_contact']} days ago")
                if ctx["open_commitments"]:
                    sections.append(f"- Open commitments ({len(ctx['open_commitments'])}):")
                    for c in ctx["open_commitments"][:3]:
                        overdue = " ⚠️ OVERDUE" if c["is_overdue"] else ""
                        sections.append(f"  • {c['description']}{overdue}")
                if ctx["risks"]:
                    sections.append(f"- Risks: {', '.join(r['title'] for r in ctx['risks'][:3])}")

        if self.company_memories:
            sections.append("## COMPANY CONTEXT")
            for cm in self.company_memories:
                ctx = cm.to_context_dict()
                sections.append(f"### {ctx['company']['name']}")
                sections.append(f"- Industry: {ctx['company']['industry']}")
                if ctx["relationship"]["days_since_contact"] is not None:
                    sections.append(f"- Last contact: {ctx['relationship']['days_since_contact']} days ago")
                if ctx["active_projects"]:
                    sections.append(f"- Active projects: {', '.join(ctx['active_projects'])}")
                if ctx["open_commitments"]:
                    sections.append(f"- Open commitments ({len(ctx['open_commitments'])}):")
                    for c in ctx["open_commitments"][:3]:
                        overdue = " ⚠️ OVERDUE" if c["is_overdue"] else ""
                        sections.append(f"  • {c['description']}{overdue}")
                if ctx["risks"]:
                    sections.append(f"- Risks ({len(ctx['risks'])} open):")
                    for r in ctx["risks"][:3]:
                        sections.append(f"  • [{r['level'].upper()}] {r['title']}")

        if self.project_memories:
            sections.append("## PROJECT CONTEXT")
            for pm in self.project_memories:
                ctx = pm.to_context_dict()
                sections.append(f"### {ctx['project']['name']} [{ctx['project']['status'].upper()}]")
                sections.append(f"- Health: {ctx['health']['assessment']} (score: {ctx['health']['score']:.2f})")
                if ctx["timeline"]["days_until_deadline"] is not None:
                    deadline_str = f"{ctx['timeline']['days_until_deadline']} days"
                    overdue = " ⚠️ OVERDUE" if ctx["timeline"]["is_overdue"] else ""
                    sections.append(f"- Deadline: {deadline_str}{overdue}")
                if ctx["open_commitments"]:
                    sections.append(f"- Open commitments ({len(ctx['open_commitments'])}):")
                    for c in ctx["open_commitments"][:3]:
                        sections.append(f"  • {c['description']} — {c.get('owner', 'unassigned')}")
                if ctx["risks"]:
                    sections.append(f"- Active risks ({len(ctx['risks'])}):")
                    for r in ctx["risks"][:3]:
                        sections.append(f"  • [{r['level'].upper()}] {r['title']}")

        if self.open_commitments:
            sections.append("## OPEN COMMITMENTS")
            for c in self.open_commitments[:15]:
                overdue = " ⚠️ OVERDUE" if c.get("is_overdue") else ""
                deadline = f" | Due: {c['deadline']}" if c.get("deadline") else ""
                sections.append(f"- {c['description']}{deadline}{overdue}")

        if self.active_risks:
            sections.append("## ACTIVE RISKS")
            for r in self.active_risks[:10]:
                sections.append(f"- [{r['level'].upper()}] {r['title']}: {r['description'][:200]}")

        if self.semantic_chunks:
            sections.append("## RELEVANT DOCUMENT EXCERPTS")
            for chunk in self.semantic_chunks[:5]:
                sections.append(f"Source: {chunk.get('source_type', 'unknown')} | {chunk.get('title', 'Untitled')}")
                sections.append(chunk.get("text", ""))

        return "\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Strategies
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Classifies query intent using keyword patterns.
    In Month 1, this is rule-based.
    Future: fine-tuned classifier.
    """

    PERSON_KEYWORDS = [
        "with", "from", "john", "pending with", "contact with",
        "last spoke", "heard from", "meeting with",
    ]
    COMPANY_KEYWORDS = [
        "schneider", "kei", "happening with", "update on", "status of",
        "relationship with",
    ]
    PROJECT_KEYWORDS = [
        "project", "rollout", "initiative", "program", "phase",
        "deployment", "implementation",
    ]
    RISK_KEYWORDS = [
        "risk", "risks", "concern", "issue", "problem", "delay",
        "blocker", "threat", "exposure",
    ]
    COMMITMENT_KEYWORDS = [
        "pending", "outstanding", "unresolved", "open", "due",
        "overdue", "promised", "commitment", "follow up",
    ]
    TIMELINE_KEYWORDS = [
        "last week", "yesterday", "this week", "recent", "lately",
        "past month", "last month",
    ]
    SUMMARY_KEYWORDS = [
        "summarize", "summary", "overview", "brief", "wrap up",
    ]

    def classify(self, query: str) -> QueryIntent:
        q = query.lower()

        if any(kw in q for kw in self.SUMMARY_KEYWORDS):
            return QueryIntent.SUMMARY_QUERY
        if any(kw in q for kw in self.RISK_KEYWORDS):
            return QueryIntent.RISK_QUERY
        if any(kw in q for kw in self.COMMITMENT_KEYWORDS):
            return QueryIntent.COMMITMENT_QUERY
        if any(kw in q for kw in self.TIMELINE_KEYWORDS):
            return QueryIntent.TIMELINE_QUERY
        if any(kw in q for kw in self.PROJECT_KEYWORDS):
            return QueryIntent.PROJECT_QUERY
        if any(kw in q for kw in self.COMPANY_KEYWORDS):
            return QueryIntent.COMPANY_QUERY
        if any(kw in q for kw in self.PERSON_KEYWORDS):
            return QueryIntent.PERSON_QUERY

        return QueryIntent.GENERAL_SEARCH


class RetrievalPlanner:
    """
    Builds a retrieval plan based on intent and resolved entities.
    The plan determines what gets fetched — before fetching begins.
    """

    def build_plan(
        self,
        query: str,
        intent: QueryIntent,
        resolved_entities: list[ResolvedEntity],
    ) -> RetrievalPlan:

        plan = RetrievalPlan(
            query=query,
            intent=intent,
            resolved_entities=resolved_entities,
        )

        if intent == QueryIntent.PERSON_QUERY:
            plan.include_person_memory = True
            plan.include_vector_search = True
            plan.vector_collection = "interactions"
            plan.vector_limit = 5
            plan.priority_signals = ["open_commitments", "last_interaction"]

        elif intent == QueryIntent.COMPANY_QUERY:
            plan.include_company_memory = True
            plan.include_person_memory = True   # Key contacts
            plan.include_vector_search = True
            plan.vector_collection = "documents"
            plan.vector_limit = 6
            plan.priority_signals = ["open_commitments", "active_projects", "risks"]

        elif intent == QueryIntent.PROJECT_QUERY:
            plan.include_project_memory = True
            plan.include_vector_search = True
            plan.vector_collection = "documents"
            plan.vector_limit = 5
            plan.priority_signals = ["health", "open_risks", "overdue_commitments"]

        elif intent == QueryIntent.RISK_QUERY:
            plan.include_open_risks = True
            plan.include_company_memory = True
            plan.include_project_memory = True
            plan.include_vector_search = True
            plan.vector_collection = "documents"
            plan.vector_limit = 4
            plan.priority_signals = ["critical_risks", "high_risks"]

        elif intent == QueryIntent.COMMITMENT_QUERY:
            plan.include_open_commitments = True
            plan.include_person_memory = False
            plan.include_vector_search = False
            plan.priority_signals = ["overdue", "upcoming_deadline"]

        elif intent == QueryIntent.TIMELINE_QUERY:
            plan.include_timeline = True
            plan.include_vector_search = True
            plan.vector_collection = "interactions"
            plan.vector_limit = 10
            plan.date_filter_days = 30

        elif intent == QueryIntent.SUMMARY_QUERY:
            plan.include_company_memory = True
            plan.include_person_memory = True
            plan.include_project_memory = True
            plan.include_vector_search = True
            plan.vector_collection = "documents"
            plan.vector_limit = 10

        else:  # GENERAL_SEARCH
            plan.include_vector_search = True
            plan.vector_collection = "documents"
            plan.vector_limit = 10

        return plan


class RetrievalEngine:
    """
    Executes retrieval plans and assembles context.

    This is the gateway before the LLM.
    Everything structured and relevant arrives here first.
    """

    def __init__(
        self,
        db: AsyncSession,
        qdrant: AsyncQdrantClient,
        embedder,                                 # EmbeddingService
    ):
        self.db = db
        self.qdrant = qdrant
        self.embedder = embedder
        self.assembler = MemoryAssembler(db)
        self.classifier = IntentClassifier()
        self.planner = RetrievalPlanner()

    async def retrieve(
        self,
        query: str,
        resolved_entities: Optional[list[ResolvedEntity]] = None,
    ) -> RetrievedContext:
        """
        Main retrieval entry point.
        Query → Intent → Plan → Fetch → Assemble → Context
        """
        if resolved_entities is None:
            resolved_entities = []

        intent = self.classifier.classify(query)
        plan = self.planner.build_plan(query, intent, resolved_entities)

        context = RetrievedContext(
            query=query,
            intent=intent,
            retrieval_plan=plan,
        )

        # ── Person memory ──────────────────────────────────────────────────
        if plan.include_person_memory:
            person_entities = [e for e in resolved_entities if e.entity_type == "person"]
            for entity in person_entities:
                if entity.entity_id:
                    pm = await self.assembler.assemble_person_memory(entity.entity_id)
                    if pm:
                        context.person_memories.append(pm)

        # ── Company memory ─────────────────────────────────────────────────
        if plan.include_company_memory:
            company_entities = [e for e in resolved_entities if e.entity_type == "company"]
            for entity in company_entities:
                if entity.entity_id:
                    cm = await self.assembler.assemble_company_memory(entity.entity_id)
                    if cm:
                        context.company_memories.append(cm)

        # ── Project memory ─────────────────────────────────────────────────
        if plan.include_project_memory:
            project_entities = [e for e in resolved_entities if e.entity_type == "project"]
            for entity in project_entities:
                if entity.entity_id:
                    pm = await self.assembler.assemble_project_memory(entity.entity_id)
                    if pm:
                        context.project_memories.append(pm)

        # ── Open commitments (global) ──────────────────────────────────────
        if plan.include_open_commitments:
            result = await self.db.execute(
                select(Commitment)
                .where(
                    Commitment.status.in_(
                        [CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS]
                    )
                )
                .order_by(Commitment.deadline.asc().nullslast())
                .limit(50)
            )
            for c in result.scalars().all():
                from app.memory.assembler import _build_commitment_memory, _days_until
                cm = _build_commitment_memory(c)
                context.open_commitments.append({
                    "id": str(c.id),
                    "description": c.description,
                    "status": c.status.value,
                    "deadline": c.deadline.isoformat() if c.deadline else None,
                    "owner": cm.owner,
                    "is_overdue": cm.is_overdue,
                    "source": c.source_type.value,
                    "project": cm.project_name,
                })

        # ── Active risks (global) ──────────────────────────────────────────
        if plan.include_open_risks:
            result = await self.db.execute(
                select(Risk)
                .where(
                    Risk.risk_status.not_in([RiskStatus.RESOLVED])
                )
                .order_by(Risk.composite_risk_score.desc())
                .limit(30)
            )
            for r in result.scalars().all():
                context.active_risks.append({
                    "id": str(r.id),
                    "title": r.title,
                    "description": r.description,
                    "level": r.risk_level.value,
                    "status": r.risk_status.value,
                    "category": r.category,
                    "score": r.composite_risk_score,
                })

        # ── Vector search ──────────────────────────────────────────────────
        if plan.include_vector_search:
            embedding = await self.embedder.embed(query)
            chunks = await self._vector_search(
                embedding=embedding,
                collection=plan.vector_collection,
                limit=plan.vector_limit,
                entities=resolved_entities,
            )
            context.semantic_chunks = chunks

        context.total_sources = (
            len(context.person_memories)
            + len(context.company_memories)
            + len(context.project_memories)
            + len(context.open_commitments)
            + len(context.active_risks)
            + len(context.semantic_chunks)
        )

        return context

    async def _vector_search(
        self,
        embedding: list[float],
        collection: str,
        limit: int,
        entities: list[ResolvedEntity],
    ) -> list[dict[str, Any]]:
        """
        Semantic search in Qdrant with optional entity filtering.
        """
        search_filter = None

        # If specific entities are resolved, constrain search to them
        if entities:
            conditions = []
            for entity in entities:
                if entity.entity_id:
                    if entity.entity_type == "company":
                        conditions.append(
                            FieldCondition(
                                key="company_id",
                                match=MatchValue(value=str(entity.entity_id)),
                            )
                        )
                    elif entity.entity_type == "person":
                        conditions.append(
                            FieldCondition(
                                key="person_id",
                                match=MatchValue(value=str(entity.entity_id)),
                            )
                        )
                    elif entity.entity_type == "project":
                        conditions.append(
                            FieldCondition(
                                key="project_id",
                                match=MatchValue(value=str(entity.entity_id)),
                            )
                        )
            if conditions:
                search_filter = Filter(should=conditions)

        results = await self.qdrant.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit,
            query_filter=search_filter,
            with_payload=True,
        )

        return [
            {
                "score": r.score,
                "text": r.payload.get("text", ""),
                "title": r.payload.get("title", ""),
                "source_type": r.payload.get("source_type", ""),
                "document_id": r.payload.get("document_id"),
                "chunk_index": r.payload.get("chunk_index"),
                "occurred_at": r.payload.get("occurred_at"),
            }
            for r in results
        ]
