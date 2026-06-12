

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import SourceAttribution, DocumentChunk
from app.core.logging import logger


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Attribution Service
# ─────────────────────────────────────────────────────────────────────────────

class AttributionService:
    """
    Records and retrieves the complete evidence chain for every memory item.

    Every time a person, company, project, relationship, event, commitment,
    or risk is created or updated — call record() to log where it came from.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def record(
        self,
        memory_item_id: UUID,
        memory_item_type: str,
        source_type: str,
        source_document_id: str,
        source_document_title: Optional[str] = None,
        source_url: Optional[str] = None,
        source_storage_path: Optional[str] = None,
        source_timestamp: Optional[datetime] = None,
        ingested_at: Optional[datetime] = None,
        participant_ids: Optional[List[UUID]] = None,
        participant_emails: Optional[List[str]] = None,
        chunk_id: Optional[UUID] = None,
        raw_excerpt: Optional[str] = None,
        excerpt_char_start: Optional[int] = None,
        excerpt_char_end: Optional[int] = None,
        confidence: float = 1.0,
    ) -> SourceAttribution:
        """
        Record provenance for a memory item.
        Safe to call multiple times — each call adds one evidence record.
        The memory item can have multiple attributions (multiple sources).
        """
        attribution = SourceAttribution(
            memory_item_id=memory_item_id,
            memory_item_type=memory_item_type,
            source_type=source_type,
            source_document_id=source_document_id,
            source_document_title=source_document_title,
            source_url=source_url,
            source_storage_path=source_storage_path,
            source_timestamp=source_timestamp,
            ingested_at=ingested_at or utcnow(),
            participant_ids=[str(p) for p in (participant_ids or [])],
            participant_emails=participant_emails or [],
            chunk_id=chunk_id,
            raw_excerpt=raw_excerpt[:2000] if raw_excerpt else None,
            excerpt_char_start=excerpt_char_start,
            excerpt_char_end=excerpt_char_end,
            confidence=confidence,
        )
        self.db.add(attribution)
        await self.db.flush()
        logger.debug(
            f"Attribution recorded: {memory_item_type}:{memory_item_id} "
            f"← {source_type}:{source_document_id}"
        )
        return attribution

    async def record_from_chunk(
        self,
        memory_item_id: UUID,
        memory_item_type: str,
        chunk: DocumentChunk,
        participant_ids: Optional[List[UUID]] = None,
        confidence: float = 1.0,
    ) -> SourceAttribution:
        """
        Convenience: record attribution directly from a DocumentChunk.
        All provenance information comes from the chunk's metadata.
        """
        return await self.record(
            memory_item_id=memory_item_id,
            memory_item_type=memory_item_type,
            source_type=chunk.source_type,
            source_document_id=chunk.document_id,
            source_timestamp=chunk.source_timestamp,
            participant_ids=participant_ids or [chunk.source_author_id] if chunk.source_author_id else [],
            chunk_id=chunk.id,
            raw_excerpt=chunk.content[:500],
            excerpt_char_start=chunk.char_start,
            excerpt_char_end=chunk.char_end,
            confidence=confidence,
        )

    # ── Provenance Retrieval ──────────────────────────────────────────────────

    async def get_provenance(
        self,
        memory_item_id: UUID,
        memory_item_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Return the complete evidence chain for a memory item.
        This is called when the CEO asks "How do you know this?"
        """
        stmt = select(SourceAttribution).where(
            and_(
                SourceAttribution.memory_item_id == memory_item_id,
                SourceAttribution.memory_item_type == memory_item_type,
            )
        ).order_by(SourceAttribution.source_timestamp.asc())

        rows = (await self.db.execute(stmt)).scalars().all()

        return [self._serialize(r) for r in rows]

    async def get_source_summary(
        self,
        memory_item_id: UUID,
        memory_item_type: str,
    ) -> Dict[str, Any]:
        """
        Returns a human-readable summary of evidence sources.
        e.g. "Based on 3 emails and 1 contract (2024-01-15 to 2024-03-20)"
        """
        attributions = await self.get_provenance(memory_item_id, memory_item_type)

        if not attributions:
            return {
                "has_attribution": False,
                "summary": "No source attribution available",
                "sources": [],
            }

        # Group by source type
        source_counts: Dict[str, int] = {}
        earliest = None
        latest = None
        participants = set()

        for a in attributions:
            st = a["source_type"]
            source_counts[st] = source_counts.get(st, 0) + 1

            ts = a.get("source_timestamp")
            if ts:
                dt = datetime.fromisoformat(ts)
                if earliest is None or dt < earliest:
                    earliest = dt
                if latest is None or dt > latest:
                    latest = dt

            participants.update(a.get("participant_emails", []))

        # Build human-readable summary
        parts = []
        for source_type, count in sorted(source_counts.items()):
            label = source_type.replace("_", " ")
            parts.append(f"{count} {label}{'s' if count > 1 else ''}")

        summary = "Based on " + ", ".join(parts)
        if earliest and latest and earliest != latest:
            summary += f" ({earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')})"
        elif earliest:
            summary += f" ({earliest.strftime('%Y-%m-%d')})"

        return {
            "has_attribution": True,
            "summary": summary,
            "source_count": len(attributions),
            "source_type_breakdown": source_counts,
            "earliest_source": earliest.isoformat() if earliest else None,
            "latest_source": latest.isoformat() if latest else None,
            "participant_emails": list(participants)[:20],
            "sources": attributions,
        }

    async def get_document_attributions(
        self,
        source_document_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Find all memory items that were produced from a specific document.
        Useful for "what did we learn from this contract?"
        """
        stmt = select(SourceAttribution).where(
            SourceAttribution.source_document_id == source_document_id
        ).order_by(SourceAttribution.memory_item_type)

        rows = (await self.db.execute(stmt)).scalars().all()
        return [self._serialize(r) for r in rows]

    async def find_unattributed_items(
        self,
        memory_item_type: str,
        item_ids: List[UUID],
    ) -> List[UUID]:
        """
        Given a list of memory item IDs, return those with NO attribution.
        Used by the validation engine (Task 9).
        """
        if not item_ids:
            return []

        stmt = select(SourceAttribution.memory_item_id).where(
            and_(
                SourceAttribution.memory_item_id.in_(item_ids),
                SourceAttribution.memory_item_type == memory_item_type,
            )
        ).distinct()

        attributed_ids = set(
            str(r) for r in (await self.db.execute(stmt)).scalars().all()
        )
        return [iid for iid in item_ids if str(iid) not in attributed_ids]

    async def get_attribution_coverage(self) -> Dict[str, Any]:
        """
        Returns attribution coverage percentage per memory item type.
        """
        stmt = select(
            SourceAttribution.memory_item_type,
            func.count(func.distinct(SourceAttribution.memory_item_id)).label("attributed_count"),
        ).group_by(SourceAttribution.memory_item_type)

        rows = (await self.db.execute(stmt)).all()

        coverage = {}
        for row in rows:
            coverage[row.memory_item_type] = row.attributed_count

        return coverage

    def _serialize(self, a: SourceAttribution) -> Dict[str, Any]:
        return {
            "id": str(a.id),
            "memory_item_id": str(a.memory_item_id),
            "memory_item_type": a.memory_item_type,
            "source_type": a.source_type,
            "source_document_id": a.source_document_id,
            "source_document_title": a.source_document_title,
            "source_url": a.source_url,
            "source_storage_path": a.source_storage_path,
            "source_timestamp": a.source_timestamp.isoformat() if a.source_timestamp else None,
            "ingested_at": a.ingested_at.isoformat() if a.ingested_at else None,
            "participant_ids": a.participant_ids or [],
            "participant_emails": a.participant_emails or [],
            "chunk_id": str(a.chunk_id) if a.chunk_id else None,
            "raw_excerpt": a.raw_excerpt,
            "confidence": a.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Attribution Decorator (utility for wrapping service calls)
# ─────────────────────────────────────────────────────────────────────────────

class AttributedOperation:
    """
    Context manager that ensures attribution is recorded for any
    memory item created within the block.

    Usage:
        async with AttributedOperation(
            attribution_service,
            source_type="email",
            source_document_id="email_abc123",
            source_timestamp=email.sent_at,
        ) as op:
            person = await person_service.get_or_create(...)
            await op.attribute(person.id, "person")
    """

    def __init__(
        self,
        attribution_service: AttributionService,
        source_type: str,
        source_document_id: str,
        source_timestamp: Optional[datetime] = None,
        source_document_title: Optional[str] = None,
        participant_ids: Optional[List[UUID]] = None,
        participant_emails: Optional[List[str]] = None,
        raw_excerpt: Optional[str] = None,
    ):
        self.svc = attribution_service
        self.source_type = source_type
        self.source_document_id = source_document_id
        self.source_timestamp = source_timestamp
        self.source_document_title = source_document_title
        self.participant_ids = participant_ids or []
        self.participant_emails = participant_emails or []
        self.raw_excerpt = raw_excerpt
        self._pending: List[tuple] = []

    async def __aenter__(self):
        return self

    async def attribute(
        self,
        item_id: UUID,
        item_type: str,
        confidence: float = 1.0,
        raw_excerpt: Optional[str] = None,
    ):
        """Queue an attribution record to be written."""
        self._pending.append((item_id, item_type, confidence, raw_excerpt))

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False  # Don't suppress exceptions

        for item_id, item_type, confidence, excerpt in self._pending:
            try:
                await self.svc.record(
                    memory_item_id=item_id,
                    memory_item_type=item_type,
                    source_type=self.source_type,
                    source_document_id=self.source_document_id,
                    source_document_title=self.source_document_title,
                    source_timestamp=self.source_timestamp,
                    participant_ids=self.participant_ids,
                    participant_emails=self.participant_emails,
                    raw_excerpt=excerpt or self.raw_excerpt,
                    confidence=confidence,
                )
            except Exception as e:
                logger.error(
                    f"Failed to record attribution for {item_type}:{item_id}: {e}"
                )
