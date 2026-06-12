

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID

from celery import chain, group, chord
from celery.utils.log import get_task_logger

from app.tasks.celery_app import celery_app

logger = get_task_logger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Master Orchestration Task
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.orchestration.process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="q_default",
)
def process_document(
    self,
    document_id: str,
    source_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Master orchestration task.
    Dispatches the full pipeline for a single document.
    Uses Celery chain to ensure ordered execution.

    Chain: extract → resolve → chunk → embed → timeline → relationships
    """
    from app.tasks.extraction import extract_entities
    from app.tasks.resolution import resolve_entities
    from app.tasks.chunking import chunk_document
    from app.tasks.embedding import embed_document_chunks
    from app.tasks.timeline import extract_timeline_events
    from app.tasks.relationships import build_relationships

    logger.info(f"Orchestrating pipeline for document: {document_id} ({source_type})")

    return run_async(_orchestrate_document(
        self, document_id, source_type, content, metadata or {}
    ))


async def _orchestrate_document(
    task,
    document_id: str,
    source_type: str,
    content: str,
    metadata: Dict[str, Any],
):
    from app.db.postgres import get_db_context
    from app.models.entities import ProcessingJob

    async with get_db_context() as db:
        # Create tracking job
        job = ProcessingJob(
            celery_task_id=task.request.id,
            job_type="full_pipeline",
            source_document_id=document_id,
            source_type=source_type,
            status="processing",
            started_at=utcnow(),
            attempt_count=1,
        )
        db.add(job)
        await db.flush()

        try:
            # Stage 1: Entity extraction
            from app.tasks.extraction import _extract_entities_async
            extraction_result = await _extract_entities_async(
                db, document_id, source_type, content, metadata
            )

            # Stage 2: Entity resolution
            from app.tasks.resolution import _resolve_entities_async
            resolution_result = await _resolve_entities_async(
                db, document_id, extraction_result
            )

            # Stage 3: Chunking (can run parallel with resolution)
            from app.tasks.chunking import _chunk_document_async
            chunk_result = await _chunk_document_async(
                db, document_id, source_type, content, metadata,
                resolution_result.get("entity_context", {})
            )

            # Stage 4: Timeline extraction
            from app.tasks.timeline import _extract_timeline_async
            timeline_result = await _extract_timeline_async(
                db, document_id, source_type, content, metadata,
                resolution_result
            )

            # Stage 5: Relationship building
            from app.tasks.relationships import _build_relationships_async
            rel_result = await _build_relationships_async(
                db, document_id, resolution_result, timeline_result
            )

            # Update job state
            job.status = "completed"
            job.completed_at = utcnow()
            job.result_summary = {
                "entities_extracted": extraction_result.get("count", 0),
                "chunks_created": chunk_result.get("chunks_created", 0),
                "events_created": timeline_result.get("events_created", 0),
                "relationships_created": rel_result.get("relationships_created", 0),
            }
            await db.flush()

            logger.info(
                f"Pipeline complete for {document_id}: "
                f"{job.result_summary}"
            )
            return job.result_summary

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = utcnow()
            await db.flush()
            logger.error(f"Pipeline failed for {document_id}: {e}", exc_info=True)

            # Retry
            try:
                raise task.retry(exc=e, countdown=60 * (2 ** task.request.retries))
            except task.MaxRetriesExceededError:
                job.status = "dead_letter"
                await db.flush()
                raise


@celery_app.task(
    name="app.tasks.orchestration.process_batch",
    bind=True,
    queue="q_default",
)
def process_batch(self, documents: list):
    """
    Process multiple documents using a Celery group (parallel).
    Each document gets its own pipeline chain.
    """
    from app.tasks.orchestration import process_document

    jobs = group(
        process_document.s(
            doc["document_id"],
            doc["source_type"],
            doc["content"],
            doc.get("metadata", {}),
        )
        for doc in documents
    )

    result = jobs.apply_async()
    logger.info(f"Dispatched batch of {len(documents)} documents")
    return {"batch_id": result.id, "document_count": len(documents)}
