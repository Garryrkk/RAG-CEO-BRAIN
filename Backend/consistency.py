

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Set, Tuple
from uuid import UUID

from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import (
    Person, Company, Project, Relationship, TimelineEvent,
    DocumentChunk, EntityAlias, SourceAttribution, MemoryConsistencyLog
)
from app.core.config import settings
from app.core.logging import logger
from app.db.qdrant import get_qdrant
from app.services.entity_resolution.resolver import normalize_company_name, normalize_text


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ValidationIssue:
    def __init__(
        self,
        issue_type: str,
        entity_type: str,
        entity_ids: List[str],
        description: str,
        severity: str = "warning",  # info | warning | error | critical
        auto_fixable: bool = False,
        fix_action: Optional[str] = None,
    ):
        self.issue_type = issue_type
        self.entity_type = entity_type
        self.entity_ids = entity_ids
        self.description = description
        self.severity = severity
        self.auto_fixable = auto_fixable
        self.fix_action = fix_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type,
            "entity_type": self.entity_type,
            "ids": self.entity_ids,
            "description": self.description,
            "severity": self.severity,
            "auto_fixable": self.auto_fixable,
            "fix_action": self.fix_action,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validation Engine
# ─────────────────────────────────────────────────────────────────────────────

class MemoryValidationEngine:
    """
    Runs comprehensive memory integrity checks and produces a
    MemoryConsistencyLog report.

    Run after every major ingestion batch and daily as a health check.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.issues: List[ValidationIssue] = []
        self.auto_fixed = 0

    async def run_full_validation(
        self, auto_fix: bool = True
    ) -> MemoryConsistencyLog:
        """
        Run all validation checks and produce a consistency log.
        auto_fix=True will apply safe automatic corrections.
        """
        run_id = uuid.uuid4()
        started_at = utcnow()

        log = MemoryConsistencyLog(
            run_id=run_id,
            started_at=started_at,
            status="running",
        )
        self.db.add(log)
        await self.db.flush()

        logger.info(f"Starting memory validation run: {run_id}")

        try:
            # ── Run all checks ────────────────────────────────────────────────
            p_count = await self._check_duplicate_persons(auto_fix)
            c_count = await self._check_duplicate_companies(auto_fix)
            pr_count = await self._check_duplicate_projects(auto_fix)

            await self._check_orphaned_entities()
            await self._check_broken_relationships(auto_fix)
            await self._check_missing_timestamps()
            await self._check_attribution_gaps()
            await self._check_embedding_gaps()
            await self._check_alias_conflicts(auto_fix)
            await self._check_email_conflicts()
            await self._check_counter_sync(auto_fix)
            await self._check_timeline_accuracy()

            # ── Populate log ──────────────────────────────────────────────────
            issue_dicts = [i.to_dict() for i in self.issues]

            dup_persons = sum(1 for i in self.issues if i.issue_type == "duplicate" and i.entity_type == "person")
            dup_companies = sum(1 for i in self.issues if i.issue_type == "duplicate" and i.entity_type == "company")
            dup_projects = sum(1 for i in self.issues if i.issue_type == "duplicate" and i.entity_type == "project")
            orphaned = sum(1 for i in self.issues if i.issue_type == "orphaned")
            broken_rels = sum(1 for i in self.issues if i.issue_type == "broken_relationship")
            missing_ts = sum(1 for i in self.issues if i.issue_type == "missing_timestamp")
            attr_gaps = sum(1 for i in self.issues if i.issue_type == "attribution_gap")
            emb_gaps = sum(1 for i in self.issues if i.issue_type == "embedding_gap")
            manual_required = sum(1 for i in self.issues if not i.auto_fixable)

            log.persons_checked = p_count
            log.companies_checked = c_count
            log.projects_checked = pr_count
            log.duplicate_persons = dup_persons
            log.duplicate_companies = dup_companies
            log.duplicate_projects = dup_projects
            log.orphaned_entities = orphaned
            log.broken_relationships = broken_rels
            log.missing_timestamps = missing_ts
            log.attribution_gaps = attr_gaps
            log.embedding_gaps = emb_gaps
            log.issues = issue_dicts
            log.auto_fixed = self.auto_fixed
            log.manual_review_required = manual_required
            log.completed_at = utcnow()
            log.status = "completed"

            total_issues = len(self.issues)
            critical = sum(1 for i in self.issues if i.severity == "critical")
            log.summary = (
                f"Validation complete. {total_issues} issues found "
                f"({critical} critical, {self.auto_fixed} auto-fixed, "
                f"{manual_required} need manual review)."
            )

            await self.db.flush()
            logger.info(log.summary)
            return log

        except Exception as e:
            log.status = "failed"
            log.summary = f"Validation failed with error: {e}"
            log.completed_at = utcnow()
            await self.db.flush()
            logger.error(f"Validation run {run_id} failed: {e}", exc_info=True)
            raise

    # ── Check: Duplicate Persons ──────────────────────────────────────────────

    async def _check_duplicate_persons(self, auto_fix: bool) -> int:
        """
        Detect persons with identical emails OR highly similar names
        under different IDs.
        """
        stmt = select(Person).where(Person.is_canonical == True)
        persons = (await self.db.execute(stmt)).scalars().all()

        # Group by primary_email
        email_map: Dict[str, List[Person]] = {}
        for p in persons:
            if p.primary_email:
                key = p.primary_email.lower()
                email_map.setdefault(key, []).append(p)

        fixed = 0
        for email, dupes in email_map.items():
            if len(dupes) > 1:
                ids = [str(d.id) for d in dupes]
                self.issues.append(ValidationIssue(
                    issue_type="duplicate",
                    entity_type="person",
                    entity_ids=ids,
                    description=f"Multiple canonical persons share email '{email}': {ids}",
                    severity="error",
                    auto_fixable=True,
                    fix_action="merge_keep_first",
                ))
                if auto_fix:
                    # Keep first (oldest), mark rest as non-canonical
                    canonical = sorted(dupes, key=lambda x: x.created_at)[0]
                    for dup in dupes:
                        if dup.id != canonical.id:
                            dup.is_canonical = False
                            logger.warning(
                                f"Auto-fixed: marked person {dup.id} "
                                f"({dup.canonical_name}) as non-canonical "
                                f"(duplicate email: {email})"
                            )
                    self.auto_fixed += 1
                    fixed += 1

        # Check for very similar names (high fuzzy score)
        from rapidfuzz import fuzz
        canonical_persons = [p for p in persons]
        name_pairs_checked: Set[Tuple[str, str]] = set()

        for i, p1 in enumerate(canonical_persons):
            for p2 in canonical_persons[i + 1:]:
                key = tuple(sorted([str(p1.id), str(p2.id)]))
                if key in name_pairs_checked:
                    continue
                name_pairs_checked.add(key)

                score = fuzz.token_sort_ratio(
                    normalize_text(p1.canonical_name),
                    normalize_text(p2.canonical_name),
                ) / 100.0

                if score >= 0.95:
                    # Very high similarity — likely duplicate
                    self.issues.append(ValidationIssue(
                        issue_type="duplicate",
                        entity_type="person",
                        entity_ids=[str(p1.id), str(p2.id)],
                        description=(
                            f"Possible duplicate persons: '{p1.canonical_name}' "
                            f"and '{p2.canonical_name}' (similarity: {score:.2f})"
                        ),
                        severity="warning",
                        auto_fixable=False,
                        fix_action="manual_merge_review",
                    ))

        await self.db.flush()
        return len(persons)

    # ── Check: Duplicate Companies ────────────────────────────────────────────

    async def _check_duplicate_companies(self, auto_fix: bool) -> int:
        stmt = select(Company).where(Company.is_canonical == True)
        companies = (await self.db.execute(stmt)).scalars().all()

        # Group by normalized name
        norm_map: Dict[str, List[Company]] = {}
        for c in companies:
            key = normalize_company_name(c.canonical_name)
            norm_map.setdefault(key, []).append(c)

        for norm_name, dupes in norm_map.items():
            if len(dupes) > 1:
                ids = [str(d.id) for d in dupes]
                self.issues.append(ValidationIssue(
                    issue_type="duplicate",
                    entity_type="company",
                    entity_ids=ids,
                    description=(
                        f"Multiple companies normalize to '{norm_name}': "
                        f"{[d.canonical_name for d in dupes]}"
                    ),
                    severity="warning",
                    auto_fixable=False,
                    fix_action="manual_merge_review",
                ))

        # Domain duplicates
        domain_map: Dict[str, List[Company]] = {}
        for c in companies:
            if c.domain:
                domain_map.setdefault(c.domain.lower(), []).append(c)

        for domain, dupes in domain_map.items():
            if len(dupes) > 1:
                self.issues.append(ValidationIssue(
                    issue_type="duplicate",
                    entity_type="company",
                    entity_ids=[str(d.id) for d in dupes],
                    description=f"Multiple companies share domain '{domain}'",
                    severity="error",
                    auto_fixable=True,
                ))
                if auto_fix:
                    canonical = sorted(dupes, key=lambda x: x.created_at)[0]
                    for dup in dupes:
                        if dup.id != canonical.id:
                            dup.is_canonical = False
                    self.auto_fixed += 1

        await self.db.flush()
        return len(companies)

    # ── Check: Duplicate Projects ─────────────────────────────────────────────

    async def _check_duplicate_projects(self, auto_fix: bool) -> int:
        stmt = select(Project).where(Project.is_canonical == True)
        projects = (await self.db.execute(stmt)).scalars().all()

        # Short code duplicates
        code_map: Dict[str, List[Project]] = {}
        for p in projects:
            if p.short_code:
                code_map.setdefault(p.short_code.upper(), []).append(p)

        for code, dupes in code_map.items():
            if len(dupes) > 1:
                self.issues.append(ValidationIssue(
                    issue_type="duplicate",
                    entity_type="project",
                    entity_ids=[str(d.id) for d in dupes],
                    description=f"Multiple projects share short code '{code}'",
                    severity="error",
                    auto_fixable=False,
                ))

        return len(projects)

    # ── Check: Orphaned Entities ──────────────────────────────────────────────

    async def _check_orphaned_entities(self):
        """
        Orphaned: canonical entity with NO relationships, NO timeline events,
        NO document chunks — effectively invisible to the system.
        """
        stmt = select(Person).where(
            Person.is_canonical == True,
            Person.email_count == 0,
            Person.meeting_count == 0,
            Person.document_count == 0,
        )
        orphaned_persons = (await self.db.execute(stmt)).scalars().all()
        for p in orphaned_persons:
            self.issues.append(ValidationIssue(
                issue_type="orphaned",
                entity_type="person",
                entity_ids=[str(p.id)],
                description=(
                    f"Person '{p.canonical_name}' ({p.id}) has no "
                    "associated emails, meetings, or documents"
                ),
                severity="info",
                auto_fixable=False,
            ))

        stmt2 = select(Company).where(
            Company.is_canonical == True,
            Company.interaction_count == 0,
            Company.email_count == 0,
            Company.project_count == 0,
        )
        orphaned_companies = (await self.db.execute(stmt2)).scalars().all()
        for c in orphaned_companies:
            self.issues.append(ValidationIssue(
                issue_type="orphaned",
                entity_type="company",
                entity_ids=[str(c.id)],
                description=f"Company '{c.canonical_name}' has no associated activity",
                severity="info",
            ))

        stmt3 = select(Project).where(
            Project.is_canonical == True,
            Project.discussion_count == 0,
            Project.participant_count == 0,
        )
        orphaned_projects = (await self.db.execute(stmt3)).scalars().all()
        for p in orphaned_projects:
            self.issues.append(ValidationIssue(
                issue_type="orphaned",
                entity_type="project",
                entity_ids=[str(p.id)],
                description=f"Project '{p.canonical_name}' has no discussions or participants",
                severity="info",
            ))

    # ── Check: Broken Relationships ───────────────────────────────────────────

    async def _check_broken_relationships(self, auto_fix: bool):
        """
        Detect relationship edges pointing to entity IDs that no longer exist.
        """
        stmt = select(Relationship).where(Relationship.is_active == True)
        rels = (await self.db.execute(stmt)).scalars().all()

        person_ids = set(
            str(r.id) for r in
            (await self.db.execute(select(Person.id))).scalars().all()
        )
        company_ids = set(
            str(r.id) for r in
            (await self.db.execute(select(Company.id))).scalars().all()
        )
        project_ids = set(
            str(r.id) for r in
            (await self.db.execute(select(Project.id))).scalars().all()
        )

        entity_sets = {
            "person": person_ids,
            "company": company_ids,
            "project": project_ids,
        }

        broken = []
        for rel in rels:
            source_valid = str(rel.source_entity_id) in entity_sets.get(
                rel.source_entity_type, set()
            )
            target_valid = str(rel.target_entity_id) in entity_sets.get(
                rel.target_entity_type, set()
            )

            if not source_valid or not target_valid:
                broken.append(rel)
                self.issues.append(ValidationIssue(
                    issue_type="broken_relationship",
                    entity_type="relationship",
                    entity_ids=[str(rel.id)],
                    description=(
                        f"Relationship {rel.id} references missing entities: "
                        f"source_valid={source_valid}, target_valid={target_valid}"
                    ),
                    severity="error",
                    auto_fixable=True,
                    fix_action="deactivate_relationship",
                ))

        if auto_fix and broken:
            for rel in broken:
                rel.is_active = False
            self.auto_fixed += len(broken)
            await self.db.flush()
            logger.warning(f"Auto-fixed: deactivated {len(broken)} broken relationships")

    # ── Check: Missing Timestamps ─────────────────────────────────────────────

    async def _check_missing_timestamps(self):
        """
        Events without occurred_at, entities without first_seen_at.
        Critical because timeline reconstruction depends on timestamps.
        """
        stmt = select(func.count(TimelineEvent.id)).where(
            TimelineEvent.occurred_at == None
        )
        count = (await self.db.execute(stmt)).scalar()
        if count > 0:
            self.issues.append(ValidationIssue(
                issue_type="missing_timestamp",
                entity_type="timeline_event",
                entity_ids=[],
                description=f"{count} timeline events are missing occurred_at timestamp",
                severity="critical",
                auto_fixable=False,
            ))

        stmt2 = select(Person).where(
            Person.is_canonical == True,
            Person.first_seen_at == None,
        )
        persons_no_ts = (await self.db.execute(stmt2)).scalars().all()
        if persons_no_ts:
            self.issues.append(ValidationIssue(
                issue_type="missing_timestamp",
                entity_type="person",
                entity_ids=[str(p.id) for p in persons_no_ts],
                description=f"{len(persons_no_ts)} persons missing first_seen_at",
                severity="warning",
                auto_fixable=True,
            ))

    # ── Check: Attribution Gaps ───────────────────────────────────────────────

    async def _check_attribution_gaps(self):
        """
        Memory items with no source evidence.
        Every claim must have a source.
        """
        attributed_ids = set(
            str(r) for r in
            (await self.db.execute(
                select(SourceAttribution.memory_item_id).distinct()
            )).scalars().all()
        )

        # Check persons
        all_persons = (await self.db.execute(select(Person.id).where(Person.is_canonical == True))).scalars().all()
        unattributed_persons = [p for p in all_persons if str(p) not in attributed_ids]
        if unattributed_persons:
            self.issues.append(ValidationIssue(
                issue_type="attribution_gap",
                entity_type="person",
                entity_ids=[str(p) for p in unattributed_persons[:50]],
                description=(
                    f"{len(unattributed_persons)} persons have no source attribution. "
                    "Cannot trace where they were discovered."
                ),
                severity="error",
                auto_fixable=False,
            ))

        # Check companies
        all_companies = (await self.db.execute(select(Company.id).where(Company.is_canonical == True))).scalars().all()
        unattributed_companies = [c for c in all_companies if str(c) not in attributed_ids]
        if unattributed_companies:
            self.issues.append(ValidationIssue(
                issue_type="attribution_gap",
                entity_type="company",
                entity_ids=[str(c) for c in unattributed_companies[:50]],
                description=f"{len(unattributed_companies)} companies have no source attribution",
                severity="error",
                auto_fixable=False,
            ))

        # Check timeline events
        all_events = (await self.db.execute(select(TimelineEvent.id))).scalars().all()
        unattributed_events = [e for e in all_events if str(e) not in attributed_ids]
        if unattributed_events:
            self.issues.append(ValidationIssue(
                issue_type="attribution_gap",
                entity_type="timeline_event",
                entity_ids=[str(e) for e in unattributed_events[:50]],
                description=f"{len(unattributed_events)} timeline events have no source attribution",
                severity="warning",
            ))

    # ── Check: Embedding Gaps ─────────────────────────────────────────────────

    async def _check_embedding_gaps(self):
        """
        Chunks in DB that haven't been embedded into Qdrant.
        This prevents these chunks from being retrieved semantically.
        """
        stmt = select(func.count(DocumentChunk.id)).where(
            DocumentChunk.is_embedded == False
        )
        pending_count = (await self.db.execute(stmt)).scalar()

        if pending_count > 0:
            total = (await self.db.execute(select(func.count(DocumentChunk.id)))).scalar()
            pct = round(pending_count / total * 100, 1) if total > 0 else 0

            severity = "critical" if pct > 20 else "warning" if pct > 5 else "info"
            self.issues.append(ValidationIssue(
                issue_type="embedding_gap",
                entity_type="document_chunk",
                entity_ids=[],
                description=(
                    f"{pending_count} chunks ({pct}% of {total} total) "
                    "not yet embedded into Qdrant"
                ),
                severity=severity,
                auto_fixable=True,
                fix_action="trigger_embedding_pipeline",
            ))
            self.db.embedding_gaps_found = pending_count  # type: ignore

    # ── Check: Alias Conflicts ────────────────────────────────────────────────

    async def _check_alias_conflicts(self, auto_fix: bool):
        """
        Same raw_value mapped to two different entities of the same type.
        This would cause resolution to non-deterministically pick one.
        """
        stmt = text("""
            SELECT raw_value, entity_type, COUNT(DISTINCT entity_id) as entity_count,
                   array_agg(DISTINCT entity_id::text) as entity_ids
            FROM entity_aliases
            GROUP BY raw_value, entity_type
            HAVING COUNT(DISTINCT entity_id) > 1
        """)
        rows = (await self.db.execute(stmt)).all()

        for row in rows:
            self.issues.append(ValidationIssue(
                issue_type="alias_conflict",
                entity_type=row.entity_type,
                entity_ids=list(row.entity_ids),
                description=(
                    f"Alias '{row.raw_value}' maps to {row.entity_count} different "
                    f"{row.entity_type} entities: {list(row.entity_ids)}"
                ),
                severity="error",
                auto_fixable=False,
                fix_action="manual_alias_review",
            ))

    # ── Check: Email Conflicts ────────────────────────────────────────────────

    async def _check_email_conflicts(self):
        """Same email address on two different canonical persons."""
        stmt = text("""
            SELECT primary_email, COUNT(*) as cnt, array_agg(id::text) as person_ids
            FROM persons
            WHERE primary_email IS NOT NULL AND is_canonical = true
            GROUP BY primary_email
            HAVING COUNT(*) > 1
        """)
        rows = (await self.db.execute(stmt)).all()
        for row in rows:
            self.issues.append(ValidationIssue(
                issue_type="email_conflict",
                entity_type="person",
                entity_ids=list(row.person_ids),
                description=(
                    f"Email '{row.primary_email}' exists on {row.cnt} "
                    "canonical persons — likely duplicates"
                ),
                severity="critical",
                auto_fixable=False,
                fix_action="manual_merge_review",
            ))

    # ── Check: Counter Sync ───────────────────────────────────────────────────

    async def _check_counter_sync(self, auto_fix: bool):
        """
        Verify that denormalized counters (email_count, meeting_count, etc.)
        match actual data counts. Drift can happen when records are deleted.
        """
        # Check chunk-based counters for persons
        stmt = text("""
            SELECT p.id, p.document_count,
                   COUNT(DISTINCT dc.document_id) as actual_count
            FROM persons p
            LEFT JOIN document_chunks dc
                ON dc.person_ids @> ARRAY[p.id::text]
            WHERE p.is_canonical = true
            GROUP BY p.id, p.document_count
            HAVING p.document_count != COUNT(DISTINCT dc.document_id)
            LIMIT 50
        """)
        rows = (await self.db.execute(stmt)).all()

        if rows:
            self.issues.append(ValidationIssue(
                issue_type="counter_drift",
                entity_type="person",
                entity_ids=[str(r.id) for r in rows],
                description=(
                    f"{len(rows)} persons have document_count out of sync "
                    "with actual chunk data"
                ),
                severity="info",
                auto_fixable=True,
                fix_action="recalculate_counters",
            ))
            if auto_fix:
                for row in rows:
                    person = await self.db.get(Person, row.id)
                    if person:
                        person.document_count = row.actual_count
                self.auto_fixed += len(rows)
                await self.db.flush()

    # ── Check: Timeline Accuracy ──────────────────────────────────────────────

    async def _check_timeline_accuracy(self):
        """
        Check for events with impossible dates (future dates, too-old dates).
        """
        future_cutoff = utcnow() + timedelta(days=30)
        ancient_cutoff = datetime(2000, 1, 1, tzinfo=timezone.utc)

        stmt = select(func.count(TimelineEvent.id)).where(
            or_(
                TimelineEvent.occurred_at > future_cutoff,
                TimelineEvent.occurred_at < ancient_cutoff,
            )
        )
        bad_date_count = (await self.db.execute(stmt)).scalar()

        if bad_date_count > 0:
            self.issues.append(ValidationIssue(
                issue_type="invalid_timestamp",
                entity_type="timeline_event",
                entity_ids=[],
                description=(
                    f"{bad_date_count} timeline events have suspicious dates "
                    "(future >30 days or before year 2000)"
                ),
                severity="warning",
                auto_fixable=False,
            ))

    # ── Report Generation ─────────────────────────────────────────────────────

    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        stmt = (
            select(MemoryConsistencyLog)
            .where(MemoryConsistencyLog.status == "completed")
            .order_by(MemoryConsistencyLog.started_at.desc())
            .limit(1)
        )
        log = (await self.db.execute(stmt)).scalar_one_or_none()
        if not log:
            return None
        return self._serialize_log(log)

    async def get_validation_history(
        self, limit: int = 20
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(MemoryConsistencyLog)
            .order_by(MemoryConsistencyLog.started_at.desc())
            .limit(limit)
        )
        logs = (await self.db.execute(stmt)).scalars().all()
        return [self._serialize_log(l) for l in logs]

    def _serialize_log(self, log: MemoryConsistencyLog) -> Dict[str, Any]:
        return {
            "run_id": str(log.run_id),
            "started_at": log.started_at.isoformat(),
            "completed_at": log.completed_at.isoformat() if log.completed_at else None,
            "status": log.status,
            "checked": {
                "persons": log.persons_checked,
                "companies": log.companies_checked,
                "projects": log.projects_checked,
                "events": log.events_checked,
                "chunks": log.chunks_checked,
            },
            "issues": {
                "duplicate_persons": log.duplicate_persons,
                "duplicate_companies": log.duplicate_companies,
                "duplicate_projects": log.duplicate_projects,
                "orphaned_entities": log.orphaned_entities,
                "broken_relationships": log.broken_relationships,
                "missing_timestamps": log.missing_timestamps,
                "attribution_gaps": log.attribution_gaps,
                "embedding_gaps": log.embedding_gaps,
                "total": len(log.issues or []),
            },
            "auto_fixed": log.auto_fixed,
            "manual_review_required": log.manual_review_required,
            "summary": log.summary,
            "detail": log.issues or [],
        }
