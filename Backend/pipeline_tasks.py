

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID

from celery.utils.log import get_task_logger

from app.tasks.celery_app import celery_app
from app.monitoring.metrics import (
    TASKS_TOTAL, TASK_DURATION, TASK_FAILURES,
    ENTITIES_CREATED, CHUNKS_CREATED, EVENTS_CREATED,
)

logger = get_task_logger(__name__)


def utcnow():
    return datetime.now(timezone.utc)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.extraction.extract_entities",
    bind=True, max_retries=3, default_retry_delay=30,
    queue="q_extraction",
)
def extract_entities(self, document_id: str, source_type: str, content: str, metadata: Dict):
    """Extract raw entity mentions from document content using Qwen3 via Ollama."""
    return run_async(_extract_entities_task(self, document_id, source_type, content, metadata))


async def _extract_entities_task(task, document_id, source_type, content, metadata):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        try:
            result = await _extract_entities_async(db, document_id, source_type, content, metadata)
            TASKS_TOTAL.labels(stage="extraction", status="success").inc()
            return result
        except Exception as e:
            TASK_FAILURES.labels(stage="extraction").inc()
            TASKS_TOTAL.labels(stage="extraction", status="failure").inc()
            raise task.retry(exc=e)


async def _extract_entities_async(db, document_id, source_type, content, metadata):
    """
    Use Qwen3 to extract named entities from document content.
    Returns structured list of found entities (persons, companies, projects).
    """
    import httpx
    from app.core.config import settings

    prompt = f"""Extract all named entities from the following {source_type} content.
Return ONLY a JSON object with this exact structure:
{{
  "persons": [
    {{"name": "Full Name", "email": "email if present or null", "role": "role if mentioned or null"}}
  ],
  "companies": [
    {{"name": "Company Name", "domain": "domain if present or null"}}
  ],
  "projects": [
    {{"name": "Project Name", "short_code": "acronym if present or null"}}
  ],
  "commitments": [
    {{"description": "what was committed", "owner": "person or company name", "due_date": "date or null"}}
  ],
  "risks": [
    {{"description": "risk description", "severity": "high/medium/low"}}
  ]
}}

Content:
{content[:4000]}
"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "{}")

    # Parse JSON from LLM response
    try:
        # Strip markdown code fences if present
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        entities = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"LLM returned non-JSON for {document_id}, using empty extraction")
        entities = {"persons": [], "companies": [], "projects": [], "commitments": [], "risks": []}

    total_found = (
        len(entities.get("persons", [])) +
        len(entities.get("companies", [])) +
        len(entities.get("projects", []))
    )
    logger.info(f"Extracted {total_found} entities from {document_id}")
    ENTITIES_CREATED.labels(entity_type="raw_mention").inc(total_found)

    return {
        "document_id": document_id,
        "source_type": source_type,
        "metadata": metadata,
        "entities": entities,
        "count": total_found,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: ENTITY RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.resolution.resolve_entities",
    bind=True, max_retries=3, default_retry_delay=30,
    queue="q_resolution",
)
def resolve_entities(self, extraction_result: Dict):
    return run_async(_resolve_task(self, extraction_result))


async def _resolve_task(task, extraction_result):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        try:
            result = await _resolve_entities_async(
                db,
                extraction_result["document_id"],
                extraction_result
            )
            TASKS_TOTAL.labels(stage="resolution", status="success").inc()
            return result
        except Exception as e:
            TASK_FAILURES.labels(stage="resolution").inc()
            raise task.retry(exc=e)


async def _resolve_entities_async(db, document_id, extraction_result):
    from app.services.entity_resolution.resolver import EntityResolutionEngine
    from app.services.knowledge.entity_memory import (
        PersonMemoryService, CompanyMemoryService, ProjectMemoryService
    )
    from app.services.attribution.source_attribution import AttributionService, AttributedOperation

    engine = EntityResolutionEngine(db)
    person_svc = PersonMemoryService(db)
    company_svc = CompanyMemoryService(db)
    project_svc = ProjectMemoryService(db)
    attr_svc = AttributionService(db)

    entities = extraction_result.get("entities", {})
    metadata = extraction_result.get("metadata", {})
    source_type = extraction_result.get("source_type", "unknown")

    resolved_persons = []
    resolved_companies = []
    resolved_projects = []

    # Resolve companies first (persons may reference them)
    for c_raw in entities.get("companies", []):
        name = c_raw.get("name")
        domain = c_raw.get("domain")
        if not name:
            continue

        result = await engine.resolve_company(raw_name=name, domain=domain)
        if result.resolved and result.entity_id:
            await engine.register_alias(result.entity_id, "company", name, "name",
                                        result.confidence, document_id, source_type)
            resolved_companies.append(str(result.entity_id))
        else:
            # Create new company
            company = await company_svc.get_or_create(
                canonical_name=name,
                domain=domain,
                source_timestamp=metadata.get("timestamp"),
            )
            await engine.register_alias(company.id, "company", name, "name",
                                        1.0, document_id, source_type)
            async with AttributedOperation(attr_svc, source_type, document_id,
                                           metadata.get("timestamp")) as op:
                await op.attribute(company.id, "company", 0.9)
            resolved_companies.append(str(company.id))
            ENTITIES_CREATED.labels(entity_type="company").inc()

    # Resolve persons
    for p_raw in entities.get("persons", []):
        name = p_raw.get("name")
        email = p_raw.get("email")
        if not name and not email:
            continue

        result = await engine.resolve_person(raw_name=name, email=email)
        if result.resolved and result.entity_id:
            if email:
                await engine.register_alias(result.entity_id, "person", email, "email",
                                            1.0, document_id, source_type)
            if name:
                await engine.register_alias(result.entity_id, "person", name, "name",
                                            result.confidence, document_id, source_type)
            resolved_persons.append(str(result.entity_id))
        else:
            company_id = UUID(resolved_companies[0]) if resolved_companies else None
            person = await person_svc.get_or_create(
                canonical_name=name or email,
                primary_email=email,
                company_id=company_id,
                job_title=p_raw.get("role"),
                source_document_id=document_id,
                source_type=source_type,
                source_timestamp=metadata.get("timestamp"),
            )
            if email:
                await engine.register_alias(person.id, "person", email, "email",
                                            1.0, document_id, source_type)
            async with AttributedOperation(attr_svc, source_type, document_id,
                                           metadata.get("timestamp")) as op:
                await op.attribute(person.id, "person", 0.9)
            resolved_persons.append(str(person.id))
            ENTITIES_CREATED.labels(entity_type="person").inc()

    # Resolve projects
    for proj_raw in entities.get("projects", []):
        name = proj_raw.get("name")
        code = proj_raw.get("short_code")
        if not name:
            continue

        result = await engine.resolve_project(raw_name=name, short_code=code)
        if result.resolved and result.entity_id:
            await engine.register_alias(result.entity_id, "project", name, "name",
                                        result.confidence, document_id, source_type)
            resolved_projects.append(str(result.entity_id))
        else:
            owner_company = UUID(resolved_companies[0]) if resolved_companies else None
            project = await project_svc.get_or_create(
                canonical_name=name,
                short_code=code,
                owner_company_id=owner_company,
                source_timestamp=metadata.get("timestamp"),
            )
            await engine.register_alias(project.id, "project", name, "name",
                                        1.0, document_id, source_type)
            async with AttributedOperation(attr_svc, source_type, document_id,
                                           metadata.get("timestamp")) as op:
                await op.attribute(project.id, "project", 0.9)
            resolved_projects.append(str(project.id))
            ENTITIES_CREATED.labels(entity_type="project").inc()

    return {
        "document_id": document_id,
        "resolved_persons": resolved_persons,
        "resolved_companies": resolved_companies,
        "resolved_projects": resolved_projects,
        "commitments": entities.get("commitments", []),
        "risks": entities.get("risks", []),
        "entity_context": {
            "person_ids": resolved_persons,
            "company_ids": resolved_companies,
            "project_ids": resolved_projects,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.chunking.chunk_document",
    bind=True, max_retries=3, default_retry_delay=30,
    queue="q_chunking",
)
def chunk_document(self, document_id: str, source_type: str, content: str,
                   metadata: Dict, entity_context: Dict):
    return run_async(_chunk_task(self, document_id, source_type, content, metadata, entity_context))


async def _chunk_task(task, document_id, source_type, content, metadata, entity_context):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        try:
            result = await _chunk_document_async(db, document_id, source_type, content, metadata, entity_context)
            TASKS_TOTAL.labels(stage="chunking", status="success").inc()
            return result
        except Exception as e:
            TASK_FAILURES.labels(stage="chunking").inc()
            raise task.retry(exc=e)


async def _chunk_document_async(db, document_id, source_type, content, metadata, entity_context):
    from app.services.chunking.strategy import chunking_router
    from app.models.entities import DocumentChunk
    import hashlib

    chunks = chunking_router.chunk(content, source_type, metadata)
    created = 0

    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()

        # Idempotency check
        from sqlalchemy import select
        from app.models.entities import DocumentChunk as DC
        exists_stmt = select(DC.id).where(
            DC.document_id == document_id,
            DC.content_hash == content_hash,
        )
        exists = (await db.execute(exists_stmt)).scalar_one_or_none()
        if exists:
            continue

        source_ts = metadata.get("timestamp")
        if isinstance(source_ts, str):
            source_ts = datetime.fromisoformat(source_ts)

        db_chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            content_hash=content_hash,
            source_type=source_type,
            chunking_strategy=chunk.chunking_strategy,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            section_title=chunk.section_title,
            person_ids=entity_context.get("person_ids", []),
            company_ids=entity_context.get("company_ids", []),
            project_ids=entity_context.get("project_ids", []),
            source_timestamp=source_ts,
            is_embedded=False,
        )
        db.add(db_chunk)
        created += 1

    await db.flush()
    CHUNKS_CREATED.inc(created)
    logger.info(f"Created {created} chunks for {document_id}")
    return {"document_id": document_id, "chunks_created": created}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.embedding.embed_document_chunks",
    bind=True, max_retries=3, default_retry_delay=120,
    queue="q_embedding",
)
def embed_document_chunks(self, document_id: str):
    return run_async(_embed_task(self, document_id))


@celery_app.task(
    name="app.tasks.embedding.embed_all_pending",
    bind=True,
    queue="q_embedding",
)
def embed_all_pending(self):
    """Scheduled task: embed all pending chunks."""
    return run_async(_embed_all_pending_task(self))


async def _embed_task(task, document_id):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        try:
            from app.services.embedding.pipeline import EmbeddingPipeline
            pipeline = EmbeddingPipeline(db)
            result = await pipeline.embed_pending_chunks(document_id=document_id)
            TASKS_TOTAL.labels(stage="embedding", status="success").inc()
            return result
        except Exception as e:
            TASK_FAILURES.labels(stage="embedding").inc()
            raise task.retry(exc=e, countdown=120)


async def _embed_all_pending_task(task):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        from app.services.embedding.pipeline import EmbeddingPipeline
        pipeline = EmbeddingPipeline(db)
        result = await pipeline.embed_pending_chunks(limit=500)
        logger.info(f"Scheduled embedding: {result}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: TIMELINE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.timeline.extract_timeline_events",
    bind=True, max_retries=3, default_retry_delay=30,
    queue="q_timeline",
)
def extract_timeline_events(self, document_id: str, source_type: str,
                             content: str, metadata: Dict, resolution_result: Dict):
    return run_async(_timeline_task(self, document_id, source_type, content, metadata, resolution_result))


async def _timeline_task(task, document_id, source_type, content, metadata, resolution_result):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        try:
            result = await _extract_timeline_async(db, document_id, source_type, content, metadata, resolution_result)
            TASKS_TOTAL.labels(stage="timeline", status="success").inc()
            return result
        except Exception as e:
            TASK_FAILURES.labels(stage="timeline").inc()
            raise task.retry(exc=e)


async def _extract_timeline_async(db, document_id, source_type, content, metadata, resolution_result):
    from app.services.timeline.reconstruction import TimelineService, detect_event_type
    from app.services.attribution.source_attribution import AttributionService, AttributedOperation

    timeline_svc = TimelineService(db)
    attr_svc = AttributionService(db)

    person_ids = [UUID(p) for p in resolution_result.get("resolved_persons", [])]
    company_ids = [UUID(c) for c in resolution_result.get("resolved_companies", [])]
    project_ids = [UUID(p) for p in resolution_result.get("resolved_projects", [])]

    source_ts = metadata.get("timestamp")
    if isinstance(source_ts, str):
        source_ts = datetime.fromisoformat(source_ts)
    if not source_ts:
        source_ts = utcnow()

    title = metadata.get("title") or metadata.get("subject") or f"{source_type.title()} event"
    event_type = detect_event_type(content[:1000], source_type)

    event = await timeline_svc.create_event(
        title=title[:512],
        occurred_at=source_ts,
        source_type=source_type,
        source_document_id=document_id,
        event_type=event_type,
        description=metadata.get("description"),
        person_ids=person_ids,
        company_ids=company_ids,
        project_ids=project_ids,
        raw_excerpt=content[:500],
        participants=metadata.get("participants", []),
    )

    async with AttributedOperation(attr_svc, source_type, document_id, source_ts) as op:
        await op.attribute(event.id, "timeline_event", 1.0, raw_excerpt=content[:300])

    EVENTS_CREATED.inc()
    return {
        "document_id": document_id,
        "events_created": 1,
        "event_id": str(event.id),
        "event_type": event_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: RELATIONSHIP BUILDING
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.relationships.build_relationships",
    bind=True, max_retries=3, default_retry_delay=30,
    queue="q_relationships",
)
def build_relationships(self, document_id: str, resolution_result: Dict, timeline_result: Dict):
    return run_async(_rel_task(self, document_id, resolution_result, timeline_result))


@celery_app.task(
    name="app.tasks.relationships.deactivate_stale",
    queue="q_relationships",
)
def deactivate_stale():
    return run_async(_deactivate_stale_task())


async def _rel_task(task, document_id, resolution_result, timeline_result):
    from app.db.postgres import get_db_context
    async with get_db_context() as db:
        try:
            result = await _build_relationships_async(db, document_id, resolution_result, timeline_result)
            TASKS_TOTAL.labels(stage="relationships", status="success").inc()
            return result
        except Exception as e:
            TASK_FAILURES.labels(stage="relationships").inc()
            raise task.retry(exc=e)


async def _build_relationships_async(db, document_id, resolution_result, timeline_result):
    from app.services.relationships.graph import RelationshipGraphService

    graph_svc = RelationshipGraphService(db)

    person_ids = [UUID(p) for p in resolution_result.get("resolved_persons", [])]
    company_ids = [UUID(c) for c in resolution_result.get("resolved_companies", [])]
    project_ids = [UUID(p) for p in resolution_result.get("resolved_projects", [])]

    source_ts = datetime.fromisoformat(timeline_result.get("occurred_at", utcnow().isoformat())) if isinstance(timeline_result.get("occurred_at"), str) else utcnow()

    created = 0

    # Person → Company
    for person_id in person_ids:
        for company_id in company_ids:
            await graph_svc.link_person_to_company(
                person_id, company_id, "works_at",
                evidence_document_id=document_id, observed_at=source_ts
            )
            created += 1

    # Person → Project
    for person_id in person_ids:
        for project_id in project_ids:
            await graph_svc.link_person_to_project(
                person_id, project_id, "participates_in",
                evidence_document_id=document_id, observed_at=source_ts
            )
            created += 1

    # Company → Project
    for company_id in company_ids:
        for project_id in project_ids:
            await graph_svc.link_company_to_project(
                company_id, project_id, "involved_in",
                evidence_document_id=document_id, observed_at=source_ts
            )
            created += 1

    # Person → Person (collaboration edges from same document)
    for i, p1 in enumerate(person_ids):
        for p2 in person_ids[i + 1:]:
            await graph_svc.link_persons(
                p1, p2, "collaborates_with",
                evidence_document_id=document_id, observed_at=source_ts
            )
            created += 1

    return {
        "document_id": document_id,
        "relationships_created": created,
    }


async def _deactivate_stale_task():
    from app.db.postgres import get_db_context
    from app.services.relationships.graph import RelationshipGraphService
    async with get_db_context() as db:
        svc = RelationshipGraphService(db)
        count = await svc.deactivate_stale_relationships(stale_days=180)
        logger.info(f"Deactivated {count} stale relationships")
        return {"deactivated": count}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TASKS (scheduled)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.validation.run_full_validation",
    bind=True,
    queue="q_validation",
)
def run_full_validation(self, auto_fix: bool = True):
    return run_async(_validation_task(self, auto_fix))


@celery_app.task(
    name="app.tasks.validation.index_health_check",
    queue="q_validation",
)
def index_health_check():
    return run_async(_health_check_task())


async def _validation_task(task, auto_fix):
    from app.db.postgres import get_db_context
    from app.services.validation.consistency import MemoryValidationEngine
    async with get_db_context() as db:
        engine = MemoryValidationEngine(db)
        log = await engine.run_full_validation(auto_fix=auto_fix)
        return {
            "run_id": str(log.run_id),
            "status": log.status,
            "summary": log.summary,
            "auto_fixed": log.auto_fixed,
        }


async def _health_check_task():
    from app.db.postgres import get_db_context
    from app.services.indexing.memory_index import MemoryIndexService
    async with get_db_context() as db:
        svc = MemoryIndexService(db)
        stats = await svc.get_index_stats()
        logger.info(f"Index health: {stats}")
        return stats
