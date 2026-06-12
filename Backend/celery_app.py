

from celery import Celery
from celery.schedules import crontab
from app.core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# Celery App
# ─────────────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "phase3",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.extraction",
        "app.tasks.resolution",
        "app.tasks.relationships",
        "app.tasks.chunking",
        "app.tasks.embedding",
        "app.tasks.timeline",
        "app.tasks.validation",
        "app.tasks.orchestration",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,     # one task at a time per worker slot
    task_track_started=True,

    # Results
    result_expires=86400,             # 24 hours
    result_persistent=True,

    # Retry defaults
    task_max_retries=settings.CELERY_MAX_RETRIES,

    # Queue routing
    task_default_queue="q_default",
    task_default_exchange="phase3",
    task_default_routing_key="default",

    task_routes={
        "app.tasks.extraction.*":    {"queue": "q_extraction"},
        "app.tasks.resolution.*":    {"queue": "q_resolution"},
        "app.tasks.relationships.*": {"queue": "q_relationships"},
        "app.tasks.chunking.*":      {"queue": "q_chunking"},
        "app.tasks.embedding.*":     {"queue": "q_embedding"},
        "app.tasks.timeline.*":      {"queue": "q_timeline"},
        "app.tasks.validation.*":    {"queue": "q_validation"},
        "app.tasks.orchestration.*": {"queue": "q_default"},
    },

    task_queues={
        "q_extraction":    {"exchange": "phase3", "routing_key": "extraction"},
        "q_resolution":    {"exchange": "phase3", "routing_key": "resolution"},
        "q_relationships": {"exchange": "phase3", "routing_key": "relationships"},
        "q_chunking":      {"exchange": "phase3", "routing_key": "chunking"},
        "q_embedding":     {"exchange": "phase3", "routing_key": "embedding"},
        "q_timeline":      {"exchange": "phase3", "routing_key": "timeline"},
        "q_validation":    {"exchange": "phase3", "routing_key": "validation"},
        "q_default":       {"exchange": "phase3", "routing_key": "default"},
    },

    # ── Celery Beat (scheduled tasks) ─────────────────────────────────────────
    beat_schedule={
        # Run validation every night at 2 AM
        "nightly-memory-validation": {
            "task": "app.tasks.validation.run_full_validation",
            "schedule": crontab(hour=2, minute=0),
            "options": {"queue": "q_validation"},
        },
        # Embed any pending chunks every 15 minutes
        "embed-pending-chunks": {
            "task": "app.tasks.embedding.embed_all_pending",
            "schedule": crontab(minute="*/15"),
            "options": {"queue": "q_embedding"},
        },
        # Deactivate stale relationships weekly
        "deactivate-stale-relationships": {
            "task": "app.tasks.relationships.deactivate_stale",
            "schedule": crontab(day_of_week="sunday", hour=3),
            "options": {"queue": "q_relationships"},
        },
        # Index health check every hour
        "index-health-check": {
            "task": "app.tasks.validation.index_health_check",
            "schedule": crontab(minute=0),
            "options": {"queue": "q_validation"},
        },
    },
)
