import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import select, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import TimelineEvent
from app.core.logging import logger


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Event type detection patterns
# ─────────────────────────────────────────────────────────────────────────────

EVENT_PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
    ("contract_signed", [
        re.compile(r"\b(contract|agreement|msa|sow|nda)\s+(signed|executed|finalized)\b", re.I),
        re.compile(r"\b(signed|executed)\s+(the\s+)?(contract|agreement)\b", re.I),
    ]),
    ("proposal_sent", [
        re.compile(r"\b(proposal|rfp|rfq|quote|bid)\s+(sent|submitted|delivered)\b", re.I),
        re.compile(r"\bsent\s+(the\s+)?(proposal|quote|bid)\b", re.I),
    ]),
    ("approval_requested", [
        re.compile(r"\b(approval|sign-?off|sign off)\s+(requested|needed|required)\b", re.I),
        re.compile(r"\bneed(s?)\s+approval\b", re.I),
    ]),
    ("approval_delayed", [
        re.compile(r"\b(approval|sign-?off)\s+(delayed|pending|held up|blocked)\b", re.I),
        re.compile(r"\bapproval\s+not\s+(received|given)\b", re.I),
        re.compile(r"\bwaiting\s+(for\s+)?(approval|sign-?off)\b", re.I),
    ]),
    ("approval_granted", [
        re.compile(r"\b(approved|approval\s+granted|approved\s+by)\b", re.I),
    ]),
    ("commitment_made", [
        re.compile(r"\b(committed|will|agreed|promised)\s+to\b", re.I),
        re.compile(r"\baction\s+item\b", re.I),
    ]),
    ("risk_identified", [
        re.compile(r"\b(risk|concern|issue|blocker|red\s+flag)\s+(identified|raised|flagged)\b", re.I),
        re.compile(r"\bat\s+risk\b", re.I),
    ]),
    ("decision_made", [
        re.compile(r"\b(decided|decision\s+made|resolved|agreed)\b", re.I),
    ]),
    ("meeting_held", [
        re.compile(r"\b(met|meeting|call|discussion|sync)\s+(held|took\s+place|occurred)\b", re.I),
        re.compile(r"\bheld\s+a\s+meeting\b", re.I),
    ]),
    ("document_created", [
        re.compile(r"\b(document|report|memo|brief|deck)\s+(created|prepared|written|drafted)\b", re.I),
    ]),
    ("project_started", [
        re.compile(r"\b(project|initiative|program)\s+(started|kicked\s+off|launched|began)\b", re.I),
        re.compile(r"\bkick-?off\b", re.I),
    ]),
    ("project_milestone", [
        re.compile(r"\bmilestone\b", re.I),
        re.compile(r"\bphase\s+\d+\s+(complete|done|delivered)\b", re.I),
    ]),
    ("project_completed", [
        re.compile(r"\b(project|initiative)\s+(completed|finished|closed|delivered)\b", re.I),
    ]),
    ("email_sent", []),   # Default for email source
]

EVENT_IMPORTANCE: Dict[str, float] = {
    "contract_signed": 0.95,
    "approval_granted": 0.85,
    "project_completed": 0.85,
    "approval_delayed": 0.80,
    "risk_identified": 0.75,
    "decision_made": 0.70,
    "project_milestone": 0.70,
    "commitment_made": 0.65,
    "proposal_sent": 0.65,
    "approval_requested": 0.60,
    "project_started": 0.60,
    "meeting_held": 0.40,
    "document_created": 0.35,
    "email_sent": 0.20,
}

MILESTONE_TYPES = {"contract_signed", "approval_granted", "project_completed",
                   "project_started", "project_milestone"}


def detect_event_type(content: str, source_type: str) -> str:
    """
    Scan content for event type keywords.
    Returns the highest-priority matched type, or a source-default.
    """
    content_lower = content.lower()
    for event_type, patterns in EVENT_PATTERNS:
        for pattern in patterns:
            if pattern.search(content_lower):
                return event_type

    # Defaults by source type
    defaults = {
        "email": "email_sent",
        "meeting_notes": "meeting_held",
        "document": "document_created",
        "contract": "contract_signed",
        "message": "email_sent",
        "note": "document_created",
    }
    return defaults.get(source_type, "document_created")


# ─────────────────────────────────────────────────────────────────────────────
# Timeline Service
# ─────────────────────────────────────────────────────────────────────────────

class TimelineService:
    """
    Creates timeline events from all source types and provides
    chronological reconstruction for any entity.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_event(
        self,
        title: str,
        occurred_at: datetime,
        source_type: str,
        source_document_id: str,
        event_type: Optional[str] = None,
        description: Optional[str] = None,
        person_ids: Optional[List[UUID]] = None,
        company_ids: Optional[List[UUID]] = None,
        project_ids: Optional[List[UUID]] = None,
        raw_excerpt: Optional[str] = None,
        participants: Optional[List[Dict]] = None,
        source_chunk_id: Optional[UUID] = None,
        importance_override: Optional[float] = None,
    ) -> TimelineEvent:
        """
        Convert any piece of information into a timeline event.
        Auto-detects event type if not provided.
        """
        detected_type = event_type or detect_event_type(
            (title or "") + " " + (description or ""), source_type
        )
        importance = importance_override or EVENT_IMPORTANCE.get(detected_type, 0.4)
        is_milestone = detected_type in MILESTONE_TYPES

        event = TimelineEvent(
            event_type=detected_type,
            title=title[:512],
            description=description,
            occurred_at=occurred_at,
            person_ids=[str(p) for p in (person_ids or [])],
            company_ids=[str(c) for c in (company_ids or [])],
            project_ids=[str(p) for p in (project_ids or [])],
            source_type=source_type,
            source_document_id=source_document_id,
            source_chunk_id=source_chunk_id,
            importance_score=importance,
            is_milestone=is_milestone,
            raw_excerpt=raw_excerpt[:2000] if raw_excerpt else None,
            participants=participants or [],
        )
        self.db.add(event)
        await self.db.flush()
        logger.debug(f"Timeline event: [{detected_type}] {title[:80]} @ {occurred_at.date()}")
        return event

    async def create_email_event(
        self,
        subject: str,
        sent_at: datetime,
        sender_id: UUID,
        recipient_ids: List[UUID],
        document_id: str,
        company_ids: Optional[List[UUID]] = None,
        project_ids: Optional[List[UUID]] = None,
        excerpt: Optional[str] = None,
    ) -> TimelineEvent:
        """Convenience method for email events."""
        all_person_ids = [sender_id] + recipient_ids
        event_type = detect_event_type(subject + " " + (excerpt or ""), "email")
        title = f"Email: {subject}" if not subject.startswith("Email") else subject
        return await self.create_event(
            title=title,
            occurred_at=sent_at,
            source_type="email",
            source_document_id=document_id,
            event_type=event_type,
            person_ids=all_person_ids,
            company_ids=company_ids,
            project_ids=project_ids,
            raw_excerpt=excerpt,
            participants=[{"person_id": str(sid), "role": "sender" if sid == sender_id else "recipient"}
                         for sid in all_person_ids],
        )

    async def create_meeting_event(
        self,
        title: str,
        held_at: datetime,
        attendee_ids: List[UUID],
        document_id: str,
        agenda_items: Optional[List[str]] = None,
        company_ids: Optional[List[UUID]] = None,
        project_ids: Optional[List[UUID]] = None,
        decisions: Optional[List[str]] = None,
        excerpt: Optional[str] = None,
    ) -> TimelineEvent:
        description_parts = []
        if agenda_items:
            description_parts.append("Agenda: " + "; ".join(agenda_items[:5]))
        if decisions:
            description_parts.append("Decisions: " + "; ".join(decisions[:3]))

        return await self.create_event(
            title=title,
            occurred_at=held_at,
            source_type="meeting_notes",
            source_document_id=document_id,
            event_type="meeting_held",
            description=" | ".join(description_parts) if description_parts else None,
            person_ids=attendee_ids,
            company_ids=company_ids,
            project_ids=project_ids,
            raw_excerpt=excerpt,
            participants=[{"person_id": str(a), "role": "attendee"} for a in attendee_ids],
            importance_override=0.45 if not decisions else 0.65,
        )

    async def create_contract_event(
        self,
        title: str,
        signed_at: datetime,
        document_id: str,
        signing_person_ids: List[UUID],
        company_ids: List[UUID],
        project_ids: Optional[List[UUID]] = None,
        excerpt: Optional[str] = None,
    ) -> TimelineEvent:
        return await self.create_event(
            title=title,
            occurred_at=signed_at,
            source_type="contract",
            source_document_id=document_id,
            event_type="contract_signed",
            person_ids=signing_person_ids,
            company_ids=company_ids,
            project_ids=project_ids,
            raw_excerpt=excerpt,
            importance_override=0.95,
        )

    # ── Query operations ─────────────────────────────────────────────────────

    async def get_entity_timeline(
        self,
        entity_id: UUID,
        entity_type: str,   # person | company | project
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        milestones_only: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct chronological activity for any entity.
        This is what answers: "What has been happening with Schneider for 6 months?"
        """
        id_str = str(entity_id)

        # Build filter based on entity type
        type_filter_map = {
            "person": TimelineEvent.person_ids,
            "company": TimelineEvent.company_ids,
            "project": TimelineEvent.project_ids,
        }

        if entity_type not in type_filter_map:
            return []

        # Use contains for JSONB array search
        field = type_filter_map[entity_type]
        stmt = select(TimelineEvent).where(field.contains([id_str]))

        if start_date:
            stmt = stmt.where(TimelineEvent.occurred_at >= start_date)
        if end_date:
            stmt = stmt.where(TimelineEvent.occurred_at <= end_date)
        if event_types:
            stmt = stmt.where(TimelineEvent.event_type.in_(event_types))
        if milestones_only:
            stmt = stmt.where(TimelineEvent.is_milestone == True)

        stmt = stmt.order_by(asc(TimelineEvent.occurred_at)).limit(limit)
        rows = (await self.db.execute(stmt)).scalars().all()

        return [self._serialize_event(r) for r in rows]

    async def get_grouped_timeline(
        self,
        entity_id: UUID,
        entity_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        group_by: str = "month",   # month | week | day
    ) -> Dict[str, Any]:
        """
        Returns timeline grouped by time period — ideal for executive view.
        e.g. "January: 3 meetings, 1 contract signed"
        """
        events = await self.get_entity_timeline(
            entity_id, entity_type, start_date, end_date
        )

        groups: Dict[str, List[Dict]] = {}
        for event in events:
            dt = datetime.fromisoformat(event["occurred_at"])
            if group_by == "month":
                key = dt.strftime("%B %Y")
            elif group_by == "week":
                key = f"Week {dt.isocalendar()[1]}, {dt.year}"
            else:
                key = dt.strftime("%Y-%m-%d")

            if key not in groups:
                groups[key] = []
            groups[key].append(event)

        # Build summary for each group
        grouped = []
        for period, period_events in groups.items():
            milestones = [e for e in period_events if e.get("is_milestone")]
            type_counts: Dict[str, int] = {}
            for e in period_events:
                type_counts[e["event_type"]] = type_counts.get(e["event_type"], 0) + 1

            grouped.append({
                "period": period,
                "event_count": len(period_events),
                "milestone_count": len(milestones),
                "type_breakdown": type_counts,
                "milestones": milestones,
                "events": period_events,
            })

        return {
            "entity_id": str(entity_id),
            "entity_type": entity_type,
            "group_by": group_by,
            "period_count": len(grouped),
            "total_events": len(events),
            "timeline": grouped,
        }

    async def get_cross_entity_timeline(
        self,
        person_ids: Optional[List[UUID]] = None,
        company_ids: Optional[List[UUID]] = None,
        project_ids: Optional[List[UUID]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Timeline spanning multiple entities simultaneously.
        Used for complex questions like "Show me all activity between
        our team and Schneider on the Smart Meter project last quarter."
        """
        conditions = []
        if person_ids:
            conditions.append(TimelineEvent.person_ids.contains(
                [str(p) for p in person_ids[:1]]   # at least one match
            ))
        if company_ids:
            conditions.append(TimelineEvent.company_ids.contains(
                [str(c) for c in company_ids[:1]]
            ))
        if project_ids:
            conditions.append(TimelineEvent.project_ids.contains(
                [str(p) for p in project_ids[:1]]
            ))

        if not conditions:
            return []

        from sqlalchemy import or_
        stmt = select(TimelineEvent).where(or_(*conditions))

        if start_date:
            stmt = stmt.where(TimelineEvent.occurred_at >= start_date)
        if end_date:
            stmt = stmt.where(TimelineEvent.occurred_at <= end_date)

        stmt = stmt.order_by(asc(TimelineEvent.occurred_at)).limit(limit)
        rows = (await self.db.execute(stmt)).scalars().all()
        return [self._serialize_event(r) for r in rows]

    def _serialize_event(self, event: TimelineEvent) -> Dict[str, Any]:
        return {
            "id": str(event.id),
            "event_type": event.event_type,
            "title": event.title,
            "description": event.description,
            "occurred_at": event.occurred_at.isoformat(),
            "occurred_at_precision": event.occurred_at_precision,
            "source_type": event.source_type,
            "source_document_id": event.source_document_id,
            "person_ids": event.person_ids or [],
            "company_ids": event.company_ids or [],
            "project_ids": event.project_ids or [],
            "participants": event.participants or [],
            "importance_score": event.importance_score,
            "is_milestone": event.is_milestone,
            "raw_excerpt": event.raw_excerpt,
        }
