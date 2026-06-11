
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from app.models.entities import (
    CommitmentStatus,
    RiskLevel,
    RiskStatus,
    SourceType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionMemory:
    """A single recorded touchpoint with summary and signals."""
    id: UUID
    source_type: SourceType
    occurred_at: datetime
    subject: Optional[str]
    summary: str
    commitments_count: int = 0
    risks_count: int = 0
    sentiment: Optional[float] = None            # -1.0 to 1.0
    source_reference: Optional[str] = None


@dataclass
class CommitmentMemory:
    """A commitment as it appears in memory — actionable and time-aware."""
    id: UUID
    description: str
    status: CommitmentStatus
    deadline: Optional[datetime]
    owner: Optional[str]                          # Person name
    source_type: SourceType
    source_excerpt: Optional[str]
    is_inbound: bool = False
    urgency_score: float = 0.5
    project_name: Optional[str] = None
    days_until_deadline: Optional[int] = None
    is_overdue: bool = False


@dataclass
class RiskMemory:
    """A risk as it appears in memory — contextualized and prioritized."""
    id: UUID
    title: str
    description: str
    risk_level: RiskLevel
    risk_status: RiskStatus
    category: Optional[str]
    composite_score: float
    identified_at: datetime
    source_type: Optional[SourceType]
    mitigation_notes: Optional[str]


@dataclass
class DocumentReference:
    """A document as referenced inside memory — pointer, not content."""
    id: UUID
    title: Optional[str]
    source_type: SourceType
    created_at: datetime
    summary: Optional[str]
    storage_path: Optional[str]


@dataclass
class TimelineEvent:
    """A single event in any entity's timeline."""
    occurred_at: datetime
    event_type: str                               # "email", "meeting", "commitment_created", etc.
    summary: str
    source_reference: Optional[str] = None
    significance_score: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# PERSON MEMORY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonMemory:
    """
    Everything the system knows about a person — assembled, not stored.

    When the CEO asks: "What is happening with John?"
    This is what gets returned, not a pile of email chunks.

    Components:
      - Identity snapshot
      - Interaction history (chronological)
      - Active commitments (theirs and to them)
      - Associated risks
      - Project involvement
      - Relationship health signals
    """

    # Identity
    person_id: UUID
    full_name: str
    email: Optional[str]
    role: str
    title: Optional[str]
    company_names: list[str] = field(default_factory=list)

    # Interaction History
    interactions: list[InteractionMemory] = field(default_factory=list)
    total_interaction_count: int = 0
    first_interaction_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    days_since_last_contact: Optional[int] = None

    # Commitments
    open_commitments: list[CommitmentMemory] = field(default_factory=list)
    overdue_commitments: list[CommitmentMemory] = field(default_factory=list)
    completed_commitments_count: int = 0

    # Risks
    associated_risks: list[RiskMemory] = field(default_factory=list)
    critical_risk_count: int = 0

    # Projects
    project_names: list[str] = field(default_factory=list)

    # Documents
    documents: list[DocumentReference] = field(default_factory=list)

    # Health Signals
    relationship_health_score: float = 0.5       # 0.0–1.0
    engagement_trend: str = "stable"             # "increasing", "decreasing", "stable"
    importance_score: float = 0.5

    # Summary (LLM-generated, cached)
    cached_summary: Optional[str] = None
    summary_generated_at: Optional[datetime] = None

    @property
    def has_urgent_items(self) -> bool:
        return len(self.overdue_commitments) > 0 or self.critical_risk_count > 0

    @property
    def attention_required(self) -> bool:
        if self.days_since_last_contact and self.days_since_last_contact > 14:
            return True
        return self.has_urgent_items

    def to_context_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for LLM context assembly."""
        return {
            "person": {
                "name": self.full_name,
                "email": self.email,
                "role": self.role,
                "title": self.title,
                "companies": self.company_names,
            },
            "relationship": {
                "total_interactions": self.total_interaction_count,
                "last_contact": self.last_interaction_at.isoformat() if self.last_interaction_at else None,
                "days_since_contact": self.days_since_last_contact,
                "health_score": self.relationship_health_score,
                "engagement_trend": self.engagement_trend,
            },
            "open_commitments": [
                {
                    "description": c.description,
                    "deadline": c.deadline.isoformat() if c.deadline else None,
                    "status": c.status.value,
                    "is_overdue": c.is_overdue,
                    "inbound": c.is_inbound,
                }
                for c in self.open_commitments
            ],
            "risks": [
                {
                    "title": r.title,
                    "level": r.risk_level.value,
                    "category": r.category,
                }
                for r in self.associated_risks
            ],
            "recent_interactions": [
                {
                    "date": i.occurred_at.isoformat(),
                    "type": i.source_type.value,
                    "subject": i.subject,
                    "summary": i.summary,
                }
                for i in self.interactions[:5]
            ],
            "projects": self.project_names,
            "attention_required": self.attention_required,
        }


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY MEMORY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompanyMemory:
    """
    Everything the system knows about a company — assembled from all sources.

    When the CEO asks: "What is happening with Schneider?"
    This is what the system assembles before calling the LLM.

    Components:
      - Company profile
      - Full timeline of interactions
      - Key contacts (ranked by interaction volume)
      - Active projects
      - Commitments (open, overdue, completed)
      - Risks
      - Document history
    """

    # Identity
    company_id: UUID
    name: str
    company_type: str
    industry: Optional[str]
    country: Optional[str]

    # Key Contacts
    key_contacts: list[PersonMemory] = field(default_factory=list)
    total_contact_count: int = 0

    # Interaction Timeline
    timeline: list[TimelineEvent] = field(default_factory=list)
    total_interaction_count: int = 0
    first_interaction_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    days_since_last_contact: Optional[int] = None

    # Decisions (extracted from interactions)
    recent_decisions: list[dict[str, Any]] = field(default_factory=list)

    # Commitments
    open_commitments: list[CommitmentMemory] = field(default_factory=list)
    overdue_commitments: list[CommitmentMemory] = field(default_factory=list)
    completed_commitments_count: int = 0

    # Risks
    associated_risks: list[RiskMemory] = field(default_factory=list)
    critical_risk_count: int = 0
    high_risk_count: int = 0

    # Projects
    active_projects: list[str] = field(default_factory=list)
    completed_projects_count: int = 0

    # Documents
    documents: list[DocumentReference] = field(default_factory=list)

    # Health Signals
    relationship_health_score: float = 0.5
    open_commitment_count: int = 0
    open_risk_count: int = 0

    # Summary (LLM-generated, cached)
    cached_summary: Optional[str] = None
    summary_generated_at: Optional[datetime] = None

    @property
    def requires_immediate_attention(self) -> bool:
        return (
            len(self.overdue_commitments) > 0
            or self.critical_risk_count > 0
            or self.relationship_health_score < 0.3
        )

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "company": {
                "name": self.name,
                "type": self.company_type,
                "industry": self.industry,
                "country": self.country,
            },
            "relationship": {
                "total_interactions": self.total_interaction_count,
                "last_contact": self.last_interaction_at.isoformat() if self.last_interaction_at else None,
                "days_since_contact": self.days_since_last_contact,
                "health_score": self.relationship_health_score,
            },
            "key_contacts": [
                {
                    "name": p.full_name,
                    "title": p.title,
                    "last_contact": p.last_interaction_at.isoformat() if p.last_interaction_at else None,
                }
                for p in self.key_contacts[:5]
            ],
            "active_projects": self.active_projects,
            "open_commitments": [
                {
                    "description": c.description,
                    "deadline": c.deadline.isoformat() if c.deadline else None,
                    "status": c.status.value,
                    "is_overdue": c.is_overdue,
                    "owner": c.owner,
                }
                for c in self.open_commitments
            ],
            "risks": [
                {
                    "title": r.title,
                    "level": r.risk_level.value,
                    "category": r.category,
                    "score": r.composite_score,
                }
                for r in self.associated_risks
            ],
            "recent_timeline": [
                {
                    "date": e.occurred_at.isoformat(),
                    "type": e.event_type,
                    "summary": e.summary,
                }
                for e in self.timeline[:10]
            ],
            "requires_attention": self.requires_immediate_attention,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT MEMORY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProjectMemory:
    """
    Everything the system knows about a project — assembled from all sources.

    When the CEO asks: "Status of Smart Meter Rollout?"
    This is the assembled context delivered to the LLM.

    Components:
      - Project snapshot
      - Timeline history
      - Stakeholders and their states
      - Open items (commitments, risks)
      - Documents
      - Health assessment
    """

    # Identity
    project_id: UUID
    name: str
    code: Optional[str]
    status: str
    priority: int
    description: Optional[str]

    # Owner
    owner_name: Optional[str]

    # Timeline
    start_date: Optional[datetime]
    target_end_date: Optional[datetime]
    actual_end_date: Optional[datetime]
    days_until_deadline: Optional[int]
    is_overdue: bool = False

    # History
    history: list[TimelineEvent] = field(default_factory=list)
    total_event_count: int = 0

    # Stakeholders
    stakeholders: list[PersonMemory] = field(default_factory=list)
    companies_involved: list[str] = field(default_factory=list)

    # Open Items
    open_commitments: list[CommitmentMemory] = field(default_factory=list)
    overdue_commitments: list[CommitmentMemory] = field(default_factory=list)
    completed_commitments_count: int = 0

    # Risks
    open_risks: list[RiskMemory] = field(default_factory=list)
    critical_risk_count: int = 0
    resolved_risks_count: int = 0

    # Documents
    documents: list[DocumentReference] = field(default_factory=list)

    # Health
    health_score: float = 0.5
    health_assessment: str = "stable"            # "healthy", "at_risk", "critical", "stable"

    # Summary (LLM-generated, cached)
    cached_summary: Optional[str] = None
    summary_generated_at: Optional[datetime] = None

    @property
    def open_item_count(self) -> int:
        return len(self.open_commitments) + len(self.open_risks)

    @property
    def is_at_risk(self) -> bool:
        return (
            len(self.overdue_commitments) > 0
            or self.critical_risk_count > 0
            or (self.is_overdue and self.status not in ("completed", "cancelled"))
        )

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "project": {
                "name": self.name,
                "code": self.code,
                "status": self.status,
                "priority": self.priority,
                "description": self.description,
                "owner": self.owner_name,
            },
            "timeline": {
                "start": self.start_date.isoformat() if self.start_date else None,
                "target_end": self.target_end_date.isoformat() if self.target_end_date else None,
                "days_until_deadline": self.days_until_deadline,
                "is_overdue": self.is_overdue,
            },
            "health": {
                "score": self.health_score,
                "assessment": self.health_assessment,
                "open_items": self.open_item_count,
                "is_at_risk": self.is_at_risk,
            },
            "stakeholders": [
                {
                    "name": p.full_name,
                    "title": p.title,
                    "open_commitments": len(p.open_commitments),
                }
                for p in self.stakeholders
            ],
            "companies": self.companies_involved,
            "open_commitments": [
                {
                    "description": c.description,
                    "deadline": c.deadline.isoformat() if c.deadline else None,
                    "owner": c.owner,
                    "is_overdue": c.is_overdue,
                }
                for c in self.open_commitments
            ],
            "risks": [
                {
                    "title": r.title,
                    "level": r.risk_level.value,
                    "score": r.composite_score,
                    "category": r.category,
                }
                for r in self.open_risks
            ],
            "recent_history": [
                {
                    "date": e.occurred_at.isoformat(),
                    "type": e.event_type,
                    "summary": e.summary,
                }
                for e in self.history[:8]
            ],
            "documents": [
                {
                    "title": d.title,
                    "type": d.source_type.value,
                    "date": d.created_at.isoformat(),
                }
                for d in self.documents[:10]
            ],
        }
