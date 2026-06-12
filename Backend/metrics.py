
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server,
)
from app.core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

registry = CollectorRegistry()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Task Metrics
# ─────────────────────────────────────────────────────────────────────────────

TASKS_TOTAL = Counter(
    "phase3_tasks_total",
    "Total number of pipeline tasks executed",
    ["stage", "status"],
    registry=registry,
)

TASK_DURATION = Histogram(
    "phase3_task_duration_seconds",
    "Pipeline task execution duration",
    ["stage"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=registry,
)

TASK_FAILURES = Counter(
    "phase3_task_failures_total",
    "Total pipeline task failures by stage",
    ["stage"],
    registry=registry,
)

TASK_RETRIES = Counter(
    "phase3_task_retries_total",
    "Total task retries",
    ["stage"],
    registry=registry,
)

DEAD_LETTER_TASKS = Counter(
    "phase3_dead_letter_tasks_total",
    "Tasks that exceeded max retries and were moved to dead letter",
    ["stage"],
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Entity Metrics
# ─────────────────────────────────────────────────────────────────────────────

ENTITIES_CREATED = Counter(
    "phase3_entities_created_total",
    "Total entities created",
    ["entity_type"],
    registry=registry,
)

ENTITY_RESOLUTIONS = Counter(
    "phase3_entity_resolutions_total",
    "Entity resolution outcomes",
    ["entity_type", "method", "result"],  # result: resolved | created | failed
    registry=registry,
)

ENTITY_MERGES = Counter(
    "phase3_entity_merges_total",
    "Entity merges (deduplication events)",
    ["entity_type"],
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chunk and Embedding Metrics
# ─────────────────────────────────────────────────────────────────────────────

CHUNKS_CREATED = Counter(
    "phase3_chunks_created_total",
    "Total document chunks created",
    registry=registry,
)

CHUNKS_EMBEDDED = Counter(
    "phase3_chunks_embedded_total",
    "Total chunks successfully embedded into Qdrant",
    registry=registry,
)

EMBEDDING_FAILURES = Counter(
    "phase3_embedding_failures_total",
    "Embedding failures by source type",
    ["source_type"],
    registry=registry,
)

EMBEDDING_DURATION = Histogram(
    "phase3_embedding_duration_seconds",
    "Time to embed a single chunk",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

EMBEDDING_BATCH_SIZE = Histogram(
    "phase3_embedding_batch_size",
    "Number of chunks embedded per batch",
    buckets=[1, 5, 10, 20, 32, 50, 100],
    registry=registry,
)

PENDING_EMBEDDINGS = Gauge(
    "phase3_pending_embeddings",
    "Number of chunks waiting to be embedded",
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Relationship Metrics
# ─────────────────────────────────────────────────────────────────────────────

RELATIONSHIPS_CREATED = Counter(
    "phase3_relationships_created_total",
    "Relationship edges created",
    ["relationship_type"],
    registry=registry,
)

RELATIONSHIPS_ACTIVE = Gauge(
    "phase3_relationships_active_total",
    "Currently active relationship edges",
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Timeline Metrics
# ─────────────────────────────────────────────────────────────────────────────

EVENTS_CREATED = Counter(
    "phase3_timeline_events_created_total",
    "Timeline events created",
    registry=registry,
)

EVENTS_BY_TYPE = Counter(
    "phase3_timeline_events_by_type_total",
    "Timeline events by event type",
    ["event_type"],
    registry=registry,
)

MILESTONES_CREATED = Counter(
    "phase3_milestones_created_total",
    "Milestone events created",
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Memory Consistency Metrics
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_ISSUES = Gauge(
    "phase3_validation_issues_total",
    "Memory consistency issues by type",
    ["issue_type", "severity"],
    registry=registry,
)

VALIDATION_AUTO_FIXED = Counter(
    "phase3_validation_auto_fixed_total",
    "Issues automatically fixed by validation engine",
    registry=registry,
)

ATTRIBUTION_GAPS = Gauge(
    "phase3_attribution_gaps",
    "Memory items without source attribution",
    ["entity_type"],
    registry=registry,
)

DUPLICATE_ENTITIES = Gauge(
    "phase3_duplicate_entities",
    "Detected duplicate entity count",
    ["entity_type"],
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Queue Depth Metrics (polled by worker)
# ─────────────────────────────────────────────────────────────────────────────

QUEUE_DEPTH = Gauge(
    "phase3_queue_depth",
    "Approximate number of messages in each Celery queue",
    ["queue_name"],
    registry=registry,
)

QUEUE_PROCESSING_TIME = Histogram(
    "phase3_queue_processing_latency_seconds",
    "Time from task enqueue to task start",
    ["queue_name"],
    buckets=[0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 300.0],
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Index / Memory State Gauges
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_COUNT = Gauge(
    "phase3_entity_count",
    "Total canonical entities in memory",
    ["entity_type"],
    registry=registry,
)

RELATIONSHIP_COUNT = Gauge(
    "phase3_relationship_count",
    "Total active relationship edges",
    registry=registry,
)

TIMELINE_EVENT_COUNT = Gauge(
    "phase3_timeline_event_count",
    "Total timeline events",
    registry=registry,
)

EMBEDDING_COVERAGE = Gauge(
    "phase3_embedding_coverage_pct",
    "Percentage of chunks with embeddings",
    registry=registry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics Collection Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def collect_db_metrics(db):
    """
    Collect current DB state metrics.
    Called by the background metrics collector every minute.
    """
    from sqlalchemy import select, func
    from app.models.entities import (
        Person, Company, Project, Relationship,
        TimelineEvent, DocumentChunk
    )

    # Entity counts
    for model, label in [(Person, "person"), (Company, "company"), (Project, "project")]:
        count = (await db.execute(
            select(func.count()).where(model.is_canonical == True)
        )).scalar()
        ENTITY_COUNT.labels(entity_type=label).set(count)

    # Relationship count
    rel_count = (await db.execute(
        select(func.count(Relationship.id)).where(Relationship.is_active == True)
    )).scalar()
    RELATIONSHIP_COUNT.set(rel_count)

    # Timeline events
    event_count = (await db.execute(select(func.count(TimelineEvent.id)))).scalar()
    TIMELINE_EVENT_COUNT.set(event_count)

    # Embedding coverage
    total_chunks = (await db.execute(select(func.count(DocumentChunk.id)))).scalar()
    embedded_chunks = (await db.execute(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.is_embedded == True)
    )).scalar()
    coverage = (embedded_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
    EMBEDDING_COVERAGE.set(coverage)
    PENDING_EMBEDDINGS.set(total_chunks - embedded_chunks)


async def collect_queue_metrics():
    """Poll Redis for Celery queue depths."""
    try:
        import redis.asyncio as aioredis
        from app.core.config import settings

        r = aioredis.from_url(settings.REDIS_URL)
        queues = [
            "q_extraction", "q_resolution", "q_relationships",
            "q_chunking", "q_embedding", "q_timeline",
            "q_validation", "q_default",
        ]
        for queue_name in queues:
            depth = await r.llen(queue_name) or 0
            QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)
        await r.aclose()
    except Exception:
        pass


def get_metrics() -> bytes:
    return generate_latest(registry)


def start_metrics_server():
    if settings.METRICS_ENABLED:
        start_http_server(settings.METRICS_PORT, registry=registry)
