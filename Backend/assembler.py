
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.memory.memory_models import (
    CommitmentMemory,
    CompanyMemory,
    DocumentReference,
    InteractionMemory,
    PersonMemory,
    ProjectMemory,
    RiskMemory,
    TimelineEvent,
)
from app.models.entities import (
    Commitment,
    CommitmentStatus,
    Company,
    Document,
    Interaction,
    Person,
    Project,
    Risk,
    RiskStatus,
)


def _days_until(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    delta = dt.replace(tzinfo=timezone.utc) - now
    return delta.days


def _days_since(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    delta = now - dt.replace(tzinfo=timezone.utc)
    return delta.days


def _build_commitment_memory(c: Commitment) -> CommitmentMemory:
    days = _days_until(c.deadline)
    return CommitmentMemory(
        id=c.id,
        description=c.description,
        status=c.status,
        deadline=c.deadline,
        owner=c.owner_person.full_name if c.owner_person else None,
        source_type=c.source_type,
        source_excerpt=c.source_excerpt,
        is_inbound=c.is_inbound,
        urgency_score=c.urgency_score,
        project_name=c.project.name if c.project else None,
        days_until_deadline=days,
        is_overdue=days is not None and days < 0,
    )


def _build_risk_memory(r: Risk) -> RiskMemory:
    return RiskMemory(
        id=r.id,
        title=r.title,
        description=r.description,
        risk_level=r.risk_level,
        risk_status=r.risk_status,
        category=r.category,
        composite_score=r.composite_risk_score,
        identified_at=r.identified_at,
        source_type=r.source_type,
        mitigation_notes=r.mitigation_notes,
    )


def _build_interaction_memory(i: Interaction) -> InteractionMemory:
    return InteractionMemory(
        id=i.id,
        source_type=i.source_type,
        occurred_at=i.occurred_at,
        subject=i.subject,
        summary=i.body_summary or "",
        commitments_count=i.commitments_extracted,
        risks_count=i.risks_extracted,
        sentiment=i.sentiment_score,
        source_reference=i.source_id,
    )


def _build_document_reference(d: Document) -> DocumentReference:
    return DocumentReference(
        id=d.id,
        title=d.title,
        source_type=d.source_type,
        created_at=d.created_at,
        summary=d.summary,
        storage_path=d.storage_path,
    )


class MemoryAssembler:
    """
    Assembles structured memory objects from relational data.

    Each assemble_* method builds a complete memory context
    for a given entity — ready to be passed to context assembly
    before the LLM is invoked.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─────────────────────────────────────────────────────────────────────────
    # Person Memory
    # ─────────────────────────────────────────────────────────────────────────

    async def assemble_person_memory(
        self,
        person_id: UUID,
        interaction_limit: int = 20,
    ) -> Optional[PersonMemory]:

        result = await self.db.execute(
            select(Person)
            .options(
                selectinload(Person.companies),
                selectinload(Person.projects),
                selectinload(Person.commitments_owned),
                selectinload(Person.commitments_assigned),
                selectinload(Person.risks_raised),
                selectinload(Person.interactions),
            )
            .where(Person.id == person_id)
        )
        person = result.scalar_one_or_none()
        if not person:
            return None

        # Interactions (most recent first)
        interactions = sorted(
            person.interactions,
            key=lambda i: i.occurred_at,
            reverse=True,
        )[:interaction_limit]

        # Commitments
        all_commitments = list(person.commitments_owned) + list(person.commitments_assigned)
        open_commitments = [
            _build_commitment_memory(c)
            for c in all_commitments
            if c.status in (CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS)
        ]
        overdue_commitments = [c for c in open_commitments if c.is_overdue]
        completed_count = sum(
            1 for c in all_commitments if c.status == CommitmentStatus.COMPLETED
        )

        # Risks
        active_risks = [
            _build_risk_memory(r)
            for r in person.risks_raised
            if r.risk_status not in (RiskStatus.RESOLVED,)
        ]

        return PersonMemory(
            person_id=person.id,
            full_name=person.full_name,
            email=person.email,
            role=person.role.value,
            title=person.title,
            company_names=[c.name for c in person.companies],
            interactions=[_build_interaction_memory(i) for i in interactions],
            total_interaction_count=person.interaction_count,
            first_interaction_at=min((i.occurred_at for i in person.interactions), default=None),
            last_interaction_at=person.last_interaction_at,
            days_since_last_contact=_days_since(person.last_interaction_at),
            open_commitments=open_commitments,
            overdue_commitments=overdue_commitments,
            completed_commitments_count=completed_count,
            associated_risks=active_risks,
            critical_risk_count=sum(1 for r in active_risks if r.risk_level.value == "critical"),
            project_names=[p.name for p in person.projects],
            relationship_health_score=person.importance_score,
            importance_score=person.importance_score,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Company Memory
    # ─────────────────────────────────────────────────────────────────────────

    async def assemble_company_memory(
        self,
        company_id: UUID,
        contact_limit: int = 10,
    ) -> Optional[CompanyMemory]:

        result = await self.db.execute(
            select(Company)
            .options(
                selectinload(Company.persons),
                selectinload(Company.projects),
                selectinload(Company.commitments),
                selectinload(Company.risks),
                selectinload(Company.interactions),
            )
            .where(Company.id == company_id)
        )
        company = result.scalar_one_or_none()
        if not company:
            return None

        # Contacts ranked by importance
        top_contacts = sorted(
            company.persons,
            key=lambda p: p.importance_score,
            reverse=True,
        )[:contact_limit]

        # Build lightweight person memories for contacts
        key_contacts = []
        for person in top_contacts:
            pm = await self.assemble_person_memory(person.id, interaction_limit=5)
            if pm:
                key_contacts.append(pm)

        # Timeline
        timeline_events = [
            TimelineEvent(
                occurred_at=i.occurred_at,
                event_type=i.source_type.value,
                summary=i.body_summary or i.subject or "",
                source_reference=i.source_id,
            )
            for i in sorted(company.interactions, key=lambda i: i.occurred_at, reverse=True)
        ]

        # Commitments
        open_commitments = [
            _build_commitment_memory(c)
            for c in company.commitments
            if c.status in (CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS)
        ]
        overdue_commitments = [c for c in open_commitments if c.is_overdue]
        completed_count = sum(
            1 for c in company.commitments if c.status == CommitmentStatus.COMPLETED
        )

        # Risks
        active_risks = [
            _build_risk_memory(r)
            for r in company.risks
            if r.risk_status != RiskStatus.RESOLVED
        ]

        return CompanyMemory(
            company_id=company.id,
            name=company.name,
            company_type=company.company_type.value,
            industry=company.industry,
            country=company.country,
            key_contacts=key_contacts,
            total_contact_count=len(company.persons),
            timeline=timeline_events[:30],
            total_interaction_count=len(company.interactions),
            last_interaction_at=company.last_interaction_at,
            days_since_last_contact=_days_since(company.last_interaction_at),
            open_commitments=open_commitments,
            overdue_commitments=overdue_commitments,
            completed_commitments_count=completed_count,
            associated_risks=active_risks,
            critical_risk_count=sum(1 for r in active_risks if r.risk_level.value == "critical"),
            high_risk_count=sum(1 for r in active_risks if r.risk_level.value == "high"),
            active_projects=[p.name for p in company.projects if p.status.value == "active"],
            completed_projects_count=sum(1 for p in company.projects if p.status.value == "completed"),
            relationship_health_score=company.relationship_health_score,
            open_commitment_count=company.open_commitment_count,
            open_risk_count=company.open_risk_count,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Project Memory
    # ─────────────────────────────────────────────────────────────────────────

    async def assemble_project_memory(
        self,
        project_id: UUID,
    ) -> Optional[ProjectMemory]:

        result = await self.db.execute(
            select(Project)
            .options(
                selectinload(Project.persons),
                selectinload(Project.companies),
                selectinload(Project.commitments),
                selectinload(Project.risks),
                selectinload(Project.documents),
                selectinload(Project.owner_person),
            )
            .where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            return None

        days_until = _days_until(project.target_end_date)
        is_overdue = (
            days_until is not None
            and days_until < 0
            and project.status.value not in ("completed", "cancelled")
        )

        # Build stakeholder memories
        stakeholders = []
        for person in project.persons[:8]:
            pm = await self.assemble_person_memory(person.id, interaction_limit=3)
            if pm:
                stakeholders.append(pm)

        # Timeline from interactions
        history_result = await self.db.execute(
            select(Interaction)
            .where(Interaction.project_id == project_id)
            .order_by(Interaction.occurred_at.desc())
            .limit(30)
        )
        interactions = history_result.scalars().all()
        history = [
            TimelineEvent(
                occurred_at=i.occurred_at,
                event_type=i.source_type.value,
                summary=i.body_summary or i.subject or "",
                source_reference=i.source_id,
            )
            for i in interactions
        ]

        # Commitments
        open_commitments = [
            _build_commitment_memory(c)
            for c in project.commitments
            if c.status in (CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS)
        ]
        overdue_commitments = [c for c in open_commitments if c.is_overdue]

        # Risks
        open_risks = [
            _build_risk_memory(r)
            for r in project.risks
            if r.risk_status != RiskStatus.RESOLVED
        ]
        critical_risk_count = sum(1 for r in open_risks if r.risk_level.value == "critical")

        # Health assessment
        if critical_risk_count > 0 or len(overdue_commitments) > 2 or is_overdue:
            health_assessment = "critical"
        elif len(overdue_commitments) > 0 or len(open_risks) > 3:
            health_assessment = "at_risk"
        elif project.health_score > 0.7:
            health_assessment = "healthy"
        else:
            health_assessment = "stable"

        return ProjectMemory(
            project_id=project.id,
            name=project.name,
            code=project.code,
            status=project.status.value,
            priority=project.priority,
            description=project.description,
            owner_name=project.owner_person.full_name if project.owner_person else None,
            start_date=project.start_date,
            target_end_date=project.target_end_date,
            actual_end_date=project.actual_end_date,
            days_until_deadline=days_until,
            is_overdue=is_overdue,
            history=history,
            total_event_count=len(interactions),
            stakeholders=stakeholders,
            companies_involved=[c.name for c in project.companies],
            open_commitments=open_commitments,
            overdue_commitments=overdue_commitments,
            completed_commitments_count=sum(
                1 for c in project.commitments if c.status == CommitmentStatus.COMPLETED
            ),
            open_risks=open_risks,
            critical_risk_count=critical_risk_count,
            resolved_risks_count=sum(
                1 for r in project.risks if r.risk_status == RiskStatus.RESOLVED
            ),
            documents=[_build_document_reference(d) for d in project.documents[:20]],
            health_score=project.health_score,
            health_assessment=health_assessment,
        )
