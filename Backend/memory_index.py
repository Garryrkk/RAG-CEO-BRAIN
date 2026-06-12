import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy import text, select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.db.qdrant import get_qdrant
from app.models.entities import (
    Person, Company, Project, TimelineEvent,
    DocumentChunk, Relationship, EntityAlias
)


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL Index Creation
# ─────────────────────────────────────────────────────────────────────────────

POSTGRES_INDEXES = [
    # ── Person indexes ────────────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_name_gin "
    "ON persons USING gin(to_tsvector('english', canonical_name))",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_email_lower "
    "ON persons (lower(primary_email)) WHERE primary_email IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_company "
    "ON persons (primary_company_id) WHERE primary_company_id IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_last_seen "
    "ON persons (last_seen_at DESC NULLS LAST)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_first_seen "
    "ON persons (first_seen_at) WHERE first_seen_at IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_emails_gin "
    "ON persons USING gin(emails)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_projects_gin "
    "ON persons USING gin(related_projects)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_commitments_gin "
    "ON persons USING gin(related_commitments)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_persons_risks_gin "
    "ON persons USING gin(related_risks)",

    # ── Company indexes ───────────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_name_gin "
    "ON companies USING gin(to_tsvector('english', canonical_name))",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_domain_lower "
    "ON companies (lower(domain)) WHERE domain IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_industry "
    "ON companies (industry) WHERE industry IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_last_seen "
    "ON companies (last_seen_at DESC NULLS LAST)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_projects_gin "
    "ON companies USING gin(projects)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_contracts_gin "
    "ON companies USING gin(contracts)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_risks_gin "
    "ON companies USING gin(risks)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_commitments_gin "
    "ON companies USING gin(commitments)",

    # ── Project indexes ───────────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_name_gin "
    "ON projects USING gin(to_tsvector('english', canonical_name))",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_short_code "
    "ON projects (upper(short_code)) WHERE short_code IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_status "
    "ON projects (status)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_owner_company "
    "ON projects (owner_company_id) WHERE owner_company_id IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_start_date "
    "ON projects (start_date) WHERE start_date IS NOT NULL",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_last_activity "
    "ON projects (last_activity_at DESC NULLS LAST)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_participants_gin "
    "ON projects USING gin(participants)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_projects_risks_gin "
    "ON projects USING gin(risks)",

    # ── Timeline event indexes ────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_occurred_at "
    "ON timeline_events (occurred_at)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_occurred_at_desc "
    "ON timeline_events (occurred_at DESC)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type "
    "ON timeline_events (event_type)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_source_type "
    "ON timeline_events (source_type)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_source_doc "
    "ON timeline_events (source_document_id)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_persons_gin "
    "ON timeline_events USING gin(person_ids)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_companies_gin "
    "ON timeline_events USING gin(company_ids)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_projects_gin "
    "ON timeline_events USING gin(project_ids)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_importance "
    "ON timeline_events (importance_score DESC)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_milestone "
    "ON timeline_events (is_milestone) WHERE is_milestone = true",

    # ── Document chunk indexes ────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_document_id "
    "ON document_chunks (document_id)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_source_type "
    "ON document_chunks (source_type)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedded "
    "ON document_chunks (is_embedded) WHERE is_embedded = false",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_persons_gin "
    "ON document_chunks USING gin(person_ids)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_companies_gin "
    "ON document_chunks USING gin(company_ids)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_projects_gin "
    "ON document_chunks USING gin(project_ids)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_content_hash "
    "ON document_chunks (content_hash)",

    # ── Relationship indexes ──────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rel_source_full "
    "ON relationships (source_entity_id, source_entity_type, relationship_type)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rel_target_full "
    "ON relationships (target_entity_id, target_entity_type, relationship_type)",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rel_active "
    "ON relationships (is_active) WHERE is_active = true",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rel_strength "
    "ON relationships (strength DESC)",

    # ── Entity alias indexes ──────────────────────────────────────────────────
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alias_raw_lower "
    "ON entity_aliases (lower(raw_value))",

    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alias_entity_type "
    "ON entity_aliases (entity_id, entity_type, alias_type)",
]


async def create_all_postgres_indexes(db: AsyncSession) -> Dict[str, Any]:
    """
    Create all PostgreSQL indexes concurrently.
    CONCURRENTLY means it won't lock tables during creation.
    Safe to run on production.
    """
    results = {"created": 0, "failed": 0, "errors": []}

    for idx_sql in POSTGRES_INDEXES:
        try:
            await db.execute(text(idx_sql))
            await db.commit()
            results["created"] += 1
            # Extract index name for logging
            idx_name = idx_sql.split("idx_")[1].split(" ")[0]
            logger.debug(f"Created index: idx_{idx_name}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(str(e))
            logger.warning(f"Index creation warning: {e}")
            await db.rollback()

    logger.info(
        f"Index creation complete: {results['created']} created, "
        f"{results['failed']} failed"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Memory Index Service
# ─────────────────────────────────────────────────────────────────────────────

class MemoryIndexService:
    """
    High-level service for querying structured memory via indexes.
    These queries bypass vector search and use direct DB lookups —
    fast, exact, structured.

    This is what powers:
      "Show me all persons at Schneider"
      "Show all active projects"
      "Find all emails about Smart Meter Rollout"
      "Show commitments owned by John Smith"
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── People Index Queries ──────────────────────────────────────────────────

    async def search_persons(
        self,
        name_query: Optional[str] = None,
        email_query: Optional[str] = None,
        company_id: Optional[UUID] = None,
        has_commitments: bool = False,
        has_risks: bool = False,
        last_seen_after: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Full-text + structured search across person memory.
        """
        stmt = select(Person).where(Person.is_canonical == True)

        if name_query:
            stmt = stmt.where(
                func.to_tsvector("english", Person.canonical_name)
                .op("@@")(func.plainto_tsquery("english", name_query))
            )
        if email_query:
            stmt = stmt.where(
                func.lower(Person.primary_email).contains(email_query.lower())
            )
        if company_id:
            stmt = stmt.where(Person.primary_company_id == company_id)
        if has_commitments:
            stmt = stmt.where(Person.commitment_count > 0)
        if has_risks:
            stmt = stmt.where(Person.risk_count > 0)
        if last_seen_after:
            stmt = stmt.where(Person.last_seen_at >= last_seen_after)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.db.execute(count_stmt)).scalar()

        stmt = stmt.order_by(desc(Person.last_seen_at)).offset(offset).limit(limit)
        persons = (await self.db.execute(stmt)).scalars().all()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [self._serialize_person(p) for p in persons],
        }

    async def get_persons_by_company(
        self, company_id: UUID, limit: int = 100
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(Person)
            .where(
                Person.primary_company_id == company_id,
                Person.is_canonical == True,
            )
            .order_by(desc(Person.last_seen_at))
            .limit(limit)
        )
        persons = (await self.db.execute(stmt)).scalars().all()
        return [self._serialize_person(p) for p in persons]

    async def get_persons_by_project(
        self, project_id: UUID
    ) -> List[Dict[str, Any]]:
        """Find all persons linked to a project via related_projects JSONB."""
        stmt = select(Person).where(
            Person.related_projects.contains(
                [{"project_id": str(project_id)}]
            ),
            Person.is_canonical == True,
        )
        persons = (await self.db.execute(stmt)).scalars().all()
        return [self._serialize_person(p) for p in persons]

    # ── Company Index Queries ─────────────────────────────────────────────────

    async def search_companies(
        self,
        name_query: Optional[str] = None,
        industry: Optional[str] = None,
        country: Optional[str] = None,
        has_active_contracts: bool = False,
        has_risks: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        stmt = select(Company).where(Company.is_canonical == True)

        if name_query:
            stmt = stmt.where(
                func.to_tsvector("english", Company.canonical_name)
                .op("@@")(func.plainto_tsquery("english", name_query))
            )
        if industry:
            stmt = stmt.where(Company.industry.ilike(f"%{industry}%"))
        if country:
            stmt = stmt.where(Company.country.ilike(f"%{country}%"))
        if has_active_contracts:
            stmt = stmt.where(Company.contract_count > 0)
        if has_risks:
            stmt = stmt.where(Company.risk_count > 0)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.db.execute(count_stmt)).scalar()

        stmt = stmt.order_by(desc(Company.last_seen_at)).offset(offset).limit(limit)
        companies = (await self.db.execute(stmt)).scalars().all()

        return {
            "total": total,
            "items": [self._serialize_company(c) for c in companies],
        }

    # ── Project Index Queries ─────────────────────────────────────────────────

    async def search_projects(
        self,
        name_query: Optional[str] = None,
        status: Optional[str] = None,
        company_id: Optional[UUID] = None,
        has_risks: bool = False,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        stmt = select(Project).where(Project.is_canonical == True)

        if name_query:
            stmt = stmt.where(
                func.to_tsvector("english", Project.canonical_name)
                .op("@@")(func.plainto_tsquery("english", name_query))
            )
        if status:
            stmt = stmt.where(Project.status == status)
        elif active_only:
            stmt = stmt.where(Project.status == "active")
        if company_id:
            stmt = stmt.where(Project.owner_company_id == company_id)
        if has_risks:
            stmt = stmt.where(Project.risk_count > 0)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.db.execute(count_stmt)).scalar()

        stmt = stmt.order_by(desc(Project.last_activity_at)).offset(offset).limit(limit)
        projects = (await self.db.execute(stmt)).scalars().all()

        return {
            "total": total,
            "items": [self._serialize_project(p) for p in projects],
        }

    # ── Source Index Queries ──────────────────────────────────────────────────

    async def get_chunks_by_source(
        self,
        source_type: str,
        document_id: Optional[str] = None,
        embedded_only: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        stmt = select(DocumentChunk).where(
            DocumentChunk.source_type == source_type
        )
        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)
        if embedded_only:
            stmt = stmt.where(DocumentChunk.is_embedded == True)
        stmt = stmt.order_by(DocumentChunk.chunk_index).limit(limit)

        chunks = (await self.db.execute(stmt)).scalars().all()
        return [
            {
                "id": str(c.id),
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "source_type": c.source_type,
                "section_title": c.section_title,
                "content_preview": c.content[:200],
                "is_embedded": c.is_embedded,
                "person_ids": c.person_ids,
                "company_ids": c.company_ids,
                "project_ids": c.project_ids,
            }
            for c in chunks
        ]

    # ── Date / Temporal Index Queries ─────────────────────────────────────────

    async def get_events_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        importance_min: float = 0.0,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(TimelineEvent)
            .where(
                TimelineEvent.occurred_at >= start_date,
                TimelineEvent.occurred_at <= end_date,
            )
        )
        if event_types:
            stmt = stmt.where(TimelineEvent.event_type.in_(event_types))
        if source_types:
            stmt = stmt.where(TimelineEvent.source_type.in_(source_types))
        if importance_min > 0:
            stmt = stmt.where(TimelineEvent.importance_score >= importance_min)

        stmt = stmt.order_by(asc(TimelineEvent.occurred_at)).limit(limit)
        events = (await self.db.execute(stmt)).scalars().all()

        return [
            {
                "id": str(e.id),
                "event_type": e.event_type,
                "title": e.title,
                "occurred_at": e.occurred_at.isoformat(),
                "source_type": e.source_type,
                "importance_score": e.importance_score,
                "is_milestone": e.is_milestone,
                "person_ids": e.person_ids,
                "company_ids": e.company_ids,
                "project_ids": e.project_ids,
            }
            for e in events
        ]

    # ── Summary / Index Health ────────────────────────────────────────────────

    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Returns counts of indexed entities and coverage metrics.
        """
        person_count = (await self.db.execute(
            select(func.count()).where(Person.is_canonical == True)
        )).scalar()

        company_count = (await self.db.execute(
            select(func.count()).where(Company.is_canonical == True)
        )).scalar()

        project_count = (await self.db.execute(
            select(func.count()).where(Project.is_canonical == True)
        )).scalar()

        event_count = (await self.db.execute(
            select(func.count(TimelineEvent.id))
        )).scalar()

        chunk_total = (await self.db.execute(
            select(func.count(DocumentChunk.id))
        )).scalar()

        chunk_embedded = (await self.db.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.is_embedded == True)
        )).scalar()

        rel_count = (await self.db.execute(
            select(func.count(Relationship.id)).where(Relationship.is_active == True)
        )).scalar()

        alias_count = (await self.db.execute(
            select(func.count(EntityAlias.id))
        )).scalar()

        embedding_coverage = (
            round(chunk_embedded / chunk_total * 100, 1) if chunk_total > 0 else 0.0
        )

        return {
            "entities": {
                "persons": person_count,
                "companies": company_count,
                "projects": project_count,
            },
            "relationships": rel_count,
            "aliases": alias_count,
            "timeline_events": event_count,
            "chunks": {
                "total": chunk_total,
                "embedded": chunk_embedded,
                "pending": chunk_total - chunk_embedded,
                "embedding_coverage_pct": embedding_coverage,
            },
        }

    async def get_commitment_index(
        self,
        entity_type: str,  # person | company | project
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Scan JSONB commitments across all entities of a given type.
        Returns a flat list of all commitments for fast review.
        """
        results = []
        if entity_type == "person":
            stmt = select(Person).where(Person.commitment_count > 0).limit(limit)
            entities = (await self.db.execute(stmt)).scalars().all()
            for e in entities:
                for c in (e.related_commitments or []):
                    results.append({
                        "owner_id": str(e.id),
                        "owner_name": e.canonical_name,
                        "owner_type": "person",
                        **c,
                    })
        elif entity_type == "company":
            stmt = select(Company).where(Company.commitment_count > 0).limit(limit)
            entities = (await self.db.execute(stmt)).scalars().all()
            for e in entities:
                for c in (e.commitments or []):
                    results.append({
                        "owner_id": str(e.id),
                        "owner_name": e.canonical_name,
                        "owner_type": "company",
                        **c,
                    })
        return results

    async def get_risk_index(
        self, entity_type: str = "all", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Aggregate all risks across entity types."""
        results = []
        if entity_type in ("all", "person"):
            stmt = select(Person).where(Person.risk_count > 0).limit(limit)
            for e in (await self.db.execute(stmt)).scalars().all():
                for r in (e.related_risks or []):
                    results.append({"owner": e.canonical_name, "type": "person", **r})
        if entity_type in ("all", "company"):
            stmt = select(Company).where(Company.risk_count > 0).limit(limit)
            for e in (await self.db.execute(stmt)).scalars().all():
                for r in (e.risks or []):
                    results.append({"owner": e.canonical_name, "type": "company", **r})
        if entity_type in ("all", "project"):
            stmt = select(Project).where(Project.risk_count > 0).limit(limit)
            for e in (await self.db.execute(stmt)).scalars().all():
                for r in (e.risks or []):
                    results.append({"owner": e.canonical_name, "type": "project", **r})
        return results

    # ── Serializers ───────────────────────────────────────────────────────────

    def _serialize_person(self, p: Person) -> Dict:
        return {
            "id": str(p.id),
            "canonical_name": p.canonical_name,
            "primary_email": p.primary_email,
            "job_title": p.job_title,
            "company_id": str(p.primary_company_id) if p.primary_company_id else None,
            "email_count": p.email_count,
            "meeting_count": p.meeting_count,
            "commitment_count": p.commitment_count,
            "risk_count": p.risk_count,
            "last_seen_at": p.last_seen_at.isoformat() if p.last_seen_at else None,
        }

    def _serialize_company(self, c: Company) -> Dict:
        return {
            "id": str(c.id),
            "canonical_name": c.canonical_name,
            "short_name": c.short_name,
            "domain": c.domain,
            "industry": c.industry,
            "contract_count": c.contract_count,
            "risk_count": c.risk_count,
            "project_count": c.project_count,
            "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
        }

    def _serialize_project(self, p: Project) -> Dict:
        return {
            "id": str(p.id),
            "canonical_name": p.canonical_name,
            "short_code": p.short_code,
            "status": p.status,
            "participant_count": p.participant_count,
            "risk_count": p.risk_count,
            "decision_count": p.decision_count,
            "last_activity_at": p.last_activity_at.isoformat() if p.last_activity_at else None,
        }
