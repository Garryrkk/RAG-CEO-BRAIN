

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.postgres import Base


# ── Helpers ───────────────────────────────────────────────────────────────────

def uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def now_utc():
    return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def updated_at():
    return Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — KNOWLEDGE ENTITY CREATION
# ─────────────────────────────────────────────────────────────────────────────

class Person(Base):
    """
    Canonical person entity.
    Accumulates: contact info, communication history, related projects,
    companies, documents, conversations, commitments, risks.
    """
    __tablename__ = "persons"

    id = uuid_pk()
    canonical_name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)

    # Contact Information
    primary_email = Column(String(320), nullable=True, index=True)
    emails = Column(JSONB, default=list)           # [{"email": ..., "source": ...}]
    phone_numbers = Column(JSONB, default=list)
    linkedin_url = Column(String(512), nullable=True)

    # Organizational context
    primary_company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    job_title = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)

    # Memory counters (denormalized for fast display)
    email_count = Column(Integer, default=0)
    meeting_count = Column(Integer, default=0)
    document_count = Column(Integer, default=0)
    commitment_count = Column(Integer, default=0)
    risk_count = Column(Integer, default=0)

    # Resolution metadata
    resolution_confidence = Column(Float, default=1.0)
    is_canonical = Column(Boolean, default=True)
    merged_from_ids = Column(JSONB, default=list)   # IDs of merged duplicate persons

    # Extended memory (JSON blobs for flexible schema)
    communication_history = Column(JSONB, default=dict)
    related_projects = Column(JSONB, default=list)
    related_companies = Column(JSONB, default=list)
    related_documents = Column(JSONB, default=list)
    related_conversations = Column(JSONB, default=list)
    related_commitments = Column(JSONB, default=list)
    related_risks = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    custom_metadata = Column(JSONB, default=dict)

    # Timestamps
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    created_at = now_utc()
    updated_at = updated_at()

    # Relationships
    aliases = relationship("EntityAlias", foreign_keys="EntityAlias.entity_id",
                           primaryjoin="and_(EntityAlias.entity_id == Person.id, "
                                       "EntityAlias.entity_type == 'person')",
                           lazy="dynamic")
    primary_company = relationship("Company", foreign_keys=[primary_company_id])

    __table_args__ = (
        Index("ix_persons_canonical_name", "canonical_name"),
        Index("ix_persons_primary_email", "primary_email"),
        Index("ix_persons_company", "primary_company_id"),
    )

    def __repr__(self):
        return f"<Person {self.canonical_name} ({self.primary_email})>"


class Company(Base):
    """
    Canonical company entity.
    Accumulates: interactions, contracts, meetings, emails,
    commitments, risks, projects.
    """
    __tablename__ = "companies"

    id = uuid_pk()
    canonical_name = Column(String(512), nullable=False)
    display_name = Column(String(512), nullable=True)
    short_name = Column(String(128), nullable=True)   # e.g. "SE" for Schneider Electric

    # Identity
    domain = Column(String(255), nullable=True, index=True)
    website = Column(String(512), nullable=True)
    industry = Column(String(255), nullable=True)
    country = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)

    # Memory counters
    interaction_count = Column(Integer, default=0)
    contract_count = Column(Integer, default=0)
    meeting_count = Column(Integer, default=0)
    email_count = Column(Integer, default=0)
    commitment_count = Column(Integer, default=0)
    risk_count = Column(Integer, default=0)
    project_count = Column(Integer, default=0)

    # Resolution
    resolution_confidence = Column(Float, default=1.0)
    is_canonical = Column(Boolean, default=True)
    merged_from_ids = Column(JSONB, default=list)

    # Extended memory
    interactions = Column(JSONB, default=list)
    contracts = Column(JSONB, default=list)
    meetings = Column(JSONB, default=list)
    emails = Column(JSONB, default=list)
    commitments = Column(JSONB, default=list)
    risks = Column(JSONB, default=list)
    projects = Column(JSONB, default=list)
    key_contacts = Column(JSONB, default=list)     # Person IDs
    tags = Column(JSONB, default=list)
    custom_metadata = Column(JSONB, default=dict)

    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    created_at = now_utc()
    updated_at = updated_at()

    __table_args__ = (
        Index("ix_companies_canonical_name", "canonical_name"),
        Index("ix_companies_domain", "domain"),
    )

    def __repr__(self):
        return f"<Company {self.canonical_name}>"


class Project(Base):
    """
    Canonical project entity.
    Accumulates: discussions, participants, deliverables,
    decisions, risks, dependencies.
    """
    __tablename__ = "projects"

    id = uuid_pk()
    canonical_name = Column(String(512), nullable=False)
    display_name = Column(String(512), nullable=True)
    short_code = Column(String(64), nullable=True)    # e.g. "SMR" for Smart Meter Rollout

    # Status
    status = Column(String(64), default="active")     # active, completed, on-hold, cancelled
    priority = Column(String(32), default="medium")   # high, medium, low

    # Ownership
    owner_person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    owner_company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)

    # Timeline
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)

    # Memory counters
    discussion_count = Column(Integer, default=0)
    participant_count = Column(Integer, default=0)
    document_count = Column(Integer, default=0)
    decision_count = Column(Integer, default=0)
    risk_count = Column(Integer, default=0)

    # Resolution
    resolution_confidence = Column(Float, default=1.0)
    is_canonical = Column(Boolean, default=True)
    merged_from_ids = Column(JSONB, default=list)

    # Extended memory
    discussions = Column(JSONB, default=list)
    participants = Column(JSONB, default=list)       # Person IDs + roles
    deliverables = Column(JSONB, default=list)
    decisions = Column(JSONB, default=list)
    risks = Column(JSONB, default=list)
    dependencies = Column(JSONB, default=list)       # Other project IDs + nature
    related_companies = Column(JSONB, default=list)
    related_documents = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    custom_metadata = Column(JSONB, default=dict)

    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    created_at = now_utc()
    updated_at = updated_at()

    owner_person = relationship("Person", foreign_keys=[owner_person_id])
    owner_company = relationship("Company", foreign_keys=[owner_company_id])

    __table_args__ = (
        Index("ix_projects_canonical_name", "canonical_name"),
        Index("ix_projects_status", "status"),
    )

    def __repr__(self):
        return f"<Project {self.canonical_name}>"


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — ENTITY RESOLUTION: Alias table
# ─────────────────────────────────────────────────────────────────────────────

class EntityAlias(Base):
    """
    Stores every raw reference that was resolved to a canonical entity.
    E.g. "J. Smith", "johnny", "john.smith@corp.com" → Person(canonical_name="John Smith")
    """
    __tablename__ = "entity_aliases"

    id = uuid_pk()
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    entity_type = Column(String(32), nullable=False)   # person | company | project
    raw_value = Column(Text, nullable=False)           # original string as seen in source
    alias_type = Column(String(32), nullable=False)    # email | name | abbreviation | nickname | domain
    confidence = Column(Float, default=1.0)
    source_document_id = Column(String(255), nullable=True)
    source_type = Column(String(64), nullable=True)
    created_at = now_utc()

    __table_args__ = (
        Index("ix_entity_aliases_entity", "entity_id", "entity_type"),
        Index("ix_entity_aliases_raw", "raw_value"),
        Index("ix_entity_aliases_type", "entity_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — RELATIONSHIP GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class Relationship(Base):
    """
    Relationship edge between any two entities.
    Relationship types:
        person_company, person_project, company_project,
        project_document, person_commitment, company_commitment,
        project_risk, person_person
    """
    __tablename__ = "relationships"

    id = uuid_pk()

    # Source entity
    source_entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    source_entity_type = Column(String(32), nullable=False)   # person | company | project

    # Target entity
    target_entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    target_entity_type = Column(String(32), nullable=False)

    # Relationship semantics
    relationship_type = Column(String(64), nullable=False)
    # e.g. "works_at", "leads", "participates_in", "owns", "committed_to",
    #       "at_risk", "depends_on", "contracted_with"
    relationship_label = Column(String(255), nullable=True)

    # Strength / confidence
    strength = Column(Float, default=1.0)
    frequency = Column(Integer, default=1)     # how many times this edge was observed
    is_active = Column(Boolean, default=True)

    # Evidence
    evidence_document_ids = Column(JSONB, default=list)
    evidence_event_ids = Column(JSONB, default=list)

    first_observed_at = Column(DateTime(timezone=True), nullable=True)
    last_observed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = now_utc()
    updated_at = updated_at()

    __table_args__ = (
        UniqueConstraint(
            "source_entity_id", "target_entity_id", "relationship_type",
            name="uq_relationship_edge"
        ),
        Index("ix_rel_source", "source_entity_id", "source_entity_type"),
        Index("ix_rel_target", "target_entity_id", "target_entity_type"),
        Index("ix_rel_type", "relationship_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — TIMELINE RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

class TimelineEvent(Base):
    """
    A discrete, timestamped event extracted from any source.
    Examples: Meeting Held, Proposal Sent, Contract Signed, Approval Delayed.
    """
    __tablename__ = "timeline_events"

    id = uuid_pk()
    event_type = Column(String(64), nullable=False)
    # e.g. meeting_held | email_sent | proposal_sent | contract_signed |
    #       approval_requested | approval_delayed | document_created |
    #       commitment_made | risk_identified | decision_made

    title = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    occurred_at_precision = Column(String(16), default="exact")  # exact | day | month

    # Entity references
    person_ids = Column(JSONB, default=list)
    company_ids = Column(JSONB, default=list)
    project_ids = Column(JSONB, default=list)

    # Source attribution
    source_type = Column(String(64), nullable=False)
    # email | meeting_notes | document | contract | message | note
    source_document_id = Column(String(255), nullable=True)
    source_chunk_id = Column(UUID(as_uuid=True), nullable=True)

    # Importance signals
    importance_score = Column(Float, default=0.5)
    is_milestone = Column(Boolean, default=False)

    # Raw content for traceability
    raw_excerpt = Column(Text, nullable=True)
    participants = Column(JSONB, default=list)

    created_at = now_utc()
    updated_at = updated_at()

    __table_args__ = (
        Index("ix_timeline_occurred_at", "occurred_at"),
        Index("ix_timeline_event_type", "event_type"),
        Index("ix_timeline_source_type", "source_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — DOCUMENT CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk(Base):
    """
    Chunked content unit ready for embedding.
    Each chunk preserves its source context and entity references.
    """
    __tablename__ = "document_chunks"

    id = uuid_pk()
    document_id = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)

    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)   # SHA-256

    # Chunking metadata
    source_type = Column(String(64), nullable=False)
    # email | document | meeting_notes | contract | message
    chunking_strategy = Column(String(64), nullable=False)
    # conversation | section | agenda | clause | sliding_window

    # Position in original document
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(512), nullable=True)

    # Entity references discovered in this chunk
    person_ids = Column(JSONB, default=list)
    company_ids = Column(JSONB, default=list)
    project_ids = Column(JSONB, default=list)

    # Embedding state
    is_embedded = Column(Boolean, default=False)
    embedding_model = Column(String(128), nullable=True)
    qdrant_vector_id = Column(String(255), nullable=True)
    embedded_at = Column(DateTime(timezone=True), nullable=True)

    # Attribution
    source_timestamp = Column(DateTime(timezone=True), nullable=True)
    source_author_id = Column(UUID(as_uuid=True), nullable=True)

    created_at = now_utc()

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        Index("ix_chunk_document", "document_id"),
        Index("ix_chunk_source_type", "source_type"),
        Index("ix_chunk_embedded", "is_embedded"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 — SOURCE ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class SourceAttribution(Base):
    """
    Every memory item must be traceable to its origin.
    This table provides the evidence chain for any claim the system makes.
    """
    __tablename__ = "source_attributions"

    id = uuid_pk()

    # What memory item this proves
    memory_item_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    memory_item_type = Column(String(64), nullable=False)
    # person | company | project | relationship | timeline_event | commitment | risk

    # Where it came from
    source_type = Column(String(64), nullable=False)
    # email | meeting_notes | document | contract | teams_message | notion_page
    source_document_id = Column(String(255), nullable=False)
    source_document_title = Column(String(512), nullable=True)
    source_url = Column(String(1024), nullable=True)
    source_storage_path = Column(String(1024), nullable=True)   # MinIO path

    # When it was recorded
    source_timestamp = Column(DateTime(timezone=True), nullable=True)
    ingested_at = Column(DateTime(timezone=True), nullable=True)

    # Who was involved
    participant_ids = Column(JSONB, default=list)
    participant_emails = Column(JSONB, default=list)

    # The specific chunk or section this came from
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id"), nullable=True)
    raw_excerpt = Column(Text, nullable=True)       # exact text that produced the memory
    excerpt_char_start = Column(Integer, nullable=True)
    excerpt_char_end = Column(Integer, nullable=True)

    confidence = Column(Float, default=1.0)
    created_at = now_utc()

    __table_args__ = (
        Index("ix_attribution_memory_item", "memory_item_id", "memory_item_type"),
        Index("ix_attribution_source_doc", "source_document_id"),
        Index("ix_attribution_source_type", "source_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 9 — MEMORY CONSISTENCY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class MemoryConsistencyLog(Base):
    """
    Records results of each validation run.
    Tracks: duplicates found, orphaned entities, broken relationships,
    missing timestamps, attribution gaps.
    """
    __tablename__ = "memory_consistency_logs"

    id = uuid_pk()
    run_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(32), default="running")   # running | completed | failed

    # Counts
    persons_checked = Column(Integer, default=0)
    companies_checked = Column(Integer, default=0)
    projects_checked = Column(Integer, default=0)
    events_checked = Column(Integer, default=0)
    chunks_checked = Column(Integer, default=0)

    # Issues found
    duplicate_persons = Column(Integer, default=0)
    duplicate_companies = Column(Integer, default=0)
    duplicate_projects = Column(Integer, default=0)
    orphaned_entities = Column(Integer, default=0)
    broken_relationships = Column(Integer, default=0)
    missing_timestamps = Column(Integer, default=0)
    attribution_gaps = Column(Integer, default=0)
    embedding_gaps = Column(Integer, default=0)

    # Detailed findings
    issues = Column(JSONB, default=list)
    # [{"type": "duplicate", "entity_type": "person", "ids": [...], "reason": "..."}]

    summary = Column(Text, nullable=True)
    auto_fixed = Column(Integer, default=0)
    manual_review_required = Column(Integer, default=0)

    created_at = now_utc()

    __table_args__ = (
        Index("ix_consistency_log_started", "started_at"),
        Index("ix_consistency_log_status", "status"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING STATE (Founder 2 — orchestration support)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingJob(Base):
    """
    Tracks the state of every item moving through the pipeline.
    Enables retry logic, monitoring, and dead-letter tracking.
    """
    __tablename__ = "processing_jobs"

    id = uuid_pk()
    celery_task_id = Column(String(255), nullable=True, index=True)
    job_type = Column(String(64), nullable=False)
    # entity_extraction | entity_resolution | relationship_creation |
    # chunking | embedding | timeline_extraction | validation

    source_document_id = Column(String(255), nullable=True, index=True)
    source_type = Column(String(64), nullable=True)
    entity_id = Column(UUID(as_uuid=True), nullable=True)

    status = Column(String(32), default="pending")
    # pending | queued | processing | completed | failed | retrying | dead_letter

    attempt_count = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)

    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    result_summary = Column(JSONB, default=dict)

    created_at = now_utc()
    updated_at = updated_at()

    __table_args__ = (
        Index("ix_job_status", "status"),
        Index("ix_job_type", "job_type"),
        Index("ix_job_source", "source_document_id"),
    )
