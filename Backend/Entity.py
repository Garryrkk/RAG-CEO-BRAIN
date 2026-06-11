
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class PersonRole(str, enum.Enum):
    EMPLOYEE = "employee"
    INVESTOR = "investor"
    REGULATOR = "regulator"
    VENDOR = "vendor"
    PARTNER = "partner"
    CUSTOMER = "customer"
    ADVISOR = "advisor"
    UNKNOWN = "unknown"


class CompanyType(str, enum.Enum):
    CLIENT = "client"
    VENDOR = "vendor"
    INVESTOR = "investor"
    GOVERNMENT = "government"
    PARTNER = "partner"
    COMPETITOR = "competitor"
    UNKNOWN = "unknown"


class ProjectStatus(str, enum.Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CommitmentStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskStatus(str, enum.Enum):
    IDENTIFIED = "identified"
    MONITORED = "monitored"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class SourceType(str, enum.Enum):
    EMAIL = "email"
    MEETING = "meeting"
    DOCUMENT = "document"
    SLACK = "slack"
    TEAMS = "teams"
    NOTION = "notion"
    MANUAL = "manual"


# ─────────────────────────────────────────────────────────────────────────────
# Association / Junction Tables
# ─────────────────────────────────────────────────────────────────────────────

person_project_association = Table(
    "person_project",
    Base.metadata,
    Column("person_id", UUID(as_uuid=True), ForeignKey("persons.id"), primary_key=True),
    Column("project_id", UUID(as_uuid=True), ForeignKey("projects.id"), primary_key=True),
    Column("role_on_project", String(100), nullable=True),
    Column("joined_at", DateTime(timezone=True), server_default=func.now()),
)

person_company_association = Table(
    "person_company",
    Base.metadata,
    Column("person_id", UUID(as_uuid=True), ForeignKey("persons.id"), primary_key=True),
    Column("company_id", UUID(as_uuid=True), ForeignKey("companies.id"), primary_key=True),
    Column("title", String(200), nullable=True),
    Column("is_primary", Boolean, default=True),
)

project_company_association = Table(
    "project_company",
    Base.metadata,
    Column("project_id", UUID(as_uuid=True), ForeignKey("projects.id"), primary_key=True),
    Column("company_id", UUID(as_uuid=True), ForeignKey("companies.id"), primary_key=True),
    Column("relationship_type", String(100), nullable=True),
)


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Person
# ─────────────────────────────────────────────────────────────────────────────

class Person(Base):
    """
    A person is anyone who interacts with the executive's world.
    The system understands them through their communications, commitments,
    and project involvement — not just their contact card.
    """
    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Identity
    full_name = Column(String(300), nullable=False, index=True)
    email = Column(String(320), unique=True, index=True, nullable=True)
    phone = Column(String(50), nullable=True)
    linkedin_url = Column(String(500), nullable=True)

    # Classification
    role = Column(Enum(PersonRole), default=PersonRole.UNKNOWN, nullable=False)
    title = Column(String(200), nullable=True)            # "CTO", "Account Manager"
    department = Column(String(200), nullable=True)

    # Context
    notes = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=list)            # ["key-contact", "decision-maker"]
    meta = Column(JSONB, default=dict)                    # Flexible extra fields

    # Importance scoring (computed by system, updated periodically)
    interaction_count = Column(Integer, default=0)
    last_interaction_at = Column(DateTime(timezone=True), nullable=True)
    importance_score = Column(Float, default=0.0)         # 0.0–1.0

    # Relationships
    companies = relationship(
        "Company",
        secondary=person_company_association,
        back_populates="persons",
    )
    projects = relationship(
        "Project",
        secondary=person_project_association,
        back_populates="persons",
    )
    commitments_owned = relationship(
        "Commitment",
        foreign_keys="Commitment.owner_person_id",
        back_populates="owner_person",
    )
    commitments_assigned = relationship(
        "Commitment",
        foreign_keys="Commitment.assigned_to_person_id",
        back_populates="assigned_to_person",
    )
    risks_raised = relationship("Risk", foreign_keys="Risk.raised_by_person_id", back_populates="raised_by")
    interactions = relationship("Interaction", back_populates="person")

    __table_args__ = (
        Index("ix_persons_name_lower", func.lower(full_name)),
        Index("ix_persons_importance", importance_score.desc()),
    )

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.full_name} role={self.role}>"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Company
# ─────────────────────────────────────────────────────────────────────────────

class Company(Base):
    """
    A company is an organizational entity in the executive's world.
    Companies have timelines, relationships, commitments, and risks.
    Not just a name in a CRM.
    """
    __tablename__ = "companies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Identity
    name = Column(String(500), nullable=False, index=True)
    legal_name = Column(String(500), nullable=True)
    domain = Column(String(253), nullable=True, index=True)   # "schneider-electric.com"
    website = Column(String(500), nullable=True)

    # Classification
    company_type = Column(Enum(CompanyType), default=CompanyType.UNKNOWN, nullable=False)
    industry = Column(String(200), nullable=True)
    country = Column(String(100), nullable=True)
    region = Column(String(200), nullable=True)

    # Context
    description = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=list)
    meta = Column(JSONB, default=dict)

    # Signals (computed)
    last_interaction_at = Column(DateTime(timezone=True), nullable=True)
    open_commitment_count = Column(Integer, default=0)
    open_risk_count = Column(Integer, default=0)
    relationship_health_score = Column(Float, default=0.5)    # 0.0–1.0

    # Relationships
    persons = relationship(
        "Person",
        secondary=person_company_association,
        back_populates="companies",
    )
    projects = relationship(
        "Project",
        secondary=project_company_association,
        back_populates="companies",
    )
    commitments = relationship("Commitment", back_populates="company")
    risks = relationship("Risk", back_populates="company")
    interactions = relationship("Interaction", back_populates="company")

    __table_args__ = (
        Index("ix_companies_name_lower", func.lower(name)),
    )

    def __repr__(self) -> str:
        return f"<Company id={self.id} name={self.name} type={self.company_type}>"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Project
# ─────────────────────────────────────────────────────────────────────────────

class Project(Base):
    """
    A project is an initiative with scope, timeline, and stakeholders.
    Everything in the executive's world connects to a project.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Identity
    name = Column(String(500), nullable=False, index=True)
    code = Column(String(50), nullable=True, unique=True)     # "SMR-2024", "INFRA-01"
    description = Column(Text, nullable=True)

    # Classification
    status = Column(Enum(ProjectStatus), default=ProjectStatus.PLANNING, nullable=False)
    priority = Column(Integer, default=3)                     # 1 (highest) – 5 (lowest)
    category = Column(String(200), nullable=True)             # "infrastructure", "regulatory"

    # Timeline
    start_date = Column(DateTime(timezone=True), nullable=True)
    target_end_date = Column(DateTime(timezone=True), nullable=True)
    actual_end_date = Column(DateTime(timezone=True), nullable=True)

    # Ownership
    owner_person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    owner_person = relationship("Person", foreign_keys=[owner_person_id])

    # Context
    notes = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=list)
    meta = Column(JSONB, default=dict)

    # Signals (computed)
    open_commitment_count = Column(Integer, default=0)
    open_risk_count = Column(Integer, default=0)
    health_score = Column(Float, default=0.5)                 # 0.0–1.0

    # Relationships
    persons = relationship(
        "Person",
        secondary=person_project_association,
        back_populates="projects",
    )
    companies = relationship(
        "Company",
        secondary=project_company_association,
        back_populates="projects",
    )
    commitments = relationship("Commitment", back_populates="project")
    risks = relationship("Risk", back_populates="project")
    documents = relationship("Document", back_populates="project")

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name} status={self.status}>"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Commitment
# ─────────────────────────────────────────────────────────────────────────────

class Commitment(Base):
    """
    A commitment is an operational object — the most time-sensitive entity.
    It represents a promise made or received, with a clear owner, deadline,
    and connection to the broader context it emerged from.

    This is NOT a task manager. This is an intelligence layer.
    """
    __tablename__ = "commitments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # The commitment itself
    description = Column(Text, nullable=False)
    original_text = Column(Text, nullable=True)              # Exact words from source

    # Status
    status = Column(Enum(CommitmentStatus), default=CommitmentStatus.OPEN, nullable=False)
    deadline = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Ownership — who made the commitment
    owner_person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    owner_person = relationship("Person", foreign_keys=[owner_person_id], back_populates="commitments_owned")

    # Assignment — who it's assigned to
    assigned_to_person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    assigned_to_person = relationship("Person", foreign_keys=[assigned_to_person_id], back_populates="commitments_assigned")

    # Source — where it was detected
    source_type = Column(Enum(SourceType), nullable=False)
    source_reference = Column(String(500), nullable=True)    # email ID, meeting ID, etc.
    source_excerpt = Column(Text, nullable=True)             # The surrounding context

    # Connections
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    project = relationship("Project", back_populates="commitments")

    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    company = relationship("Company", back_populates="commitments")

    # Metadata
    is_inbound = Column(Boolean, default=False)              # Did someone commit TO the executive?
    urgency_score = Column(Float, default=0.5)               # 0.0–1.0
    tags = Column(ARRAY(String), default=list)
    meta = Column(JSONB, default=dict)

    __table_args__ = (
        Index("ix_commitments_status_deadline", status, deadline),
        Index("ix_commitments_open", status),
    )

    def __repr__(self) -> str:
        return f"<Commitment id={self.id} status={self.status} deadline={self.deadline}>"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Risk
# ─────────────────────────────────────────────────────────────────────────────

class Risk(Base):
    """
    A risk is a signal — detected from communications, documents, or patterns.
    Risks connect to every other entity and form the early-warning layer.
    """
    __tablename__ = "risks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Identity
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)

    # Classification
    risk_level = Column(Enum(RiskLevel), default=RiskLevel.MEDIUM, nullable=False)
    risk_status = Column(Enum(RiskStatus), default=RiskStatus.IDENTIFIED, nullable=False)
    category = Column(String(200), nullable=True)            # "vendor", "regulatory", "timeline"

    # Detection
    detection_method = Column(String(200), nullable=True)    # "nlp_pattern", "deadline_breach", "manual"
    source_type = Column(Enum(SourceType), nullable=True)
    source_reference = Column(String(500), nullable=True)
    source_excerpt = Column(Text, nullable=True)

    # Ownership
    raised_by_person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    raised_by = relationship("Person", foreign_keys=[raised_by_person_id], back_populates="risks_raised")

    # Connections
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    project = relationship("Project", back_populates="risks")

    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    company = relationship("Company", back_populates="risks")

    # Timeline
    identified_at = Column(DateTime(timezone=True), server_default=func.now())
    last_reviewed_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Scoring
    probability_score = Column(Float, default=0.5)           # 0.0–1.0
    impact_score = Column(Float, default=0.5)                # 0.0–1.0
    composite_risk_score = Column(Float, default=0.25)       # probability × impact

    # Context
    mitigation_notes = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=list)
    meta = Column(JSONB, default=dict)

    __table_args__ = (
        Index("ix_risks_level_status", risk_level, risk_status),
        Index("ix_risks_composite_score", composite_risk_score.desc()),
    )

    def __repr__(self) -> str:
        return f"<Risk id={self.id} level={self.risk_level} status={self.risk_status}>"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Interaction
# ─────────────────────────────────────────────────────────────────────────────

class Interaction(Base):
    """
    An interaction is a recorded touchpoint — email, meeting, message.
    Interactions populate the memory layer for every entity.
    """
    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Classification
    source_type = Column(Enum(SourceType), nullable=False)
    source_id = Column(String(500), nullable=True)           # External message/meeting ID
    subject = Column(String(1000), nullable=True)
    body_summary = Column(Text, nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False)

    # Participants
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    person = relationship("Person", back_populates="interactions")

    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    company = relationship("Company", back_populates="interactions")

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)

    # Signals extracted
    commitments_extracted = Column(Integer, default=0)
    risks_extracted = Column(Integer, default=0)
    sentiment_score = Column(Float, nullable=True)           # -1.0 to 1.0

    meta = Column(JSONB, default=dict)

    __table_args__ = (
        Index("ix_interactions_occurred_at", occurred_at.desc()),
        Index("ix_interactions_person_occurred", person_id, occurred_at.desc()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY: Document
# ─────────────────────────────────────────────────────────────────────────────

class Document(Base):
    """
    A document is evidence — not the primary unit of intelligence.
    Documents are ingested, chunked, embedded, and linked to entities.
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Identity
    title = Column(String(1000), nullable=True)
    filename = Column(String(500), nullable=True)
    file_extension = Column(String(20), nullable=True)
    mime_type = Column(String(100), nullable=True)

    # Source
    source_type = Column(Enum(SourceType), nullable=False)
    source_url = Column(String(2000), nullable=True)
    source_id = Column(String(500), nullable=True)           # External ID

    # Storage
    storage_path = Column(String(2000), nullable=True)       # MinIO path
    file_size_bytes = Column(Integer, nullable=True)
    checksum = Column(String(64), nullable=True)

    # Processing
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    chunk_count = Column(Integer, default=0)
    processing_error = Column(Text, nullable=True)

    # Connections
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    project = relationship("Project", back_populates="documents")

    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)

    # Content
    summary = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=list)
    meta = Column(JSONB, default=dict)

    def __repr__(self) -> str:
        return f"<Document id={self.id} title={self.title} source={self.source_type}>"
