

import asyncio
import json
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.canonical import CanonicalDocument
from app.models.sync import SyncJob, SyncLogEntry, SyncStatus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and config
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    OUTLOOK = "outlook"
    NOTION = "notion"
    OBSIDIAN = "obsidian"
    GOOGLE_DRIVE = "google_drive"
    TEAMS = "teams"
    SLACK = "slack"
    FILE_UPLOAD = "file_upload"


class SyncMode(str, Enum):
    FULL = "full"          # Re-sync everything from scratch
    INCREMENTAL = "incremental"  # Only changes since last sync
    RESUME = "resume"      # Continue an interrupted sync


class ConnectorStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    UNHEALTHY = "unhealthy"


@dataclass
class ConnectorConfig:
    connector_id: str
    source_type: SourceType
    user_id: str
    credentials: Dict[str, Any]
    settings: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    # Retry configuration
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # Exponential backoff multiplier
    retry_backoff_max: float = 300.0  # Max 5 minutes between retries

    # Rate limiting
    requests_per_second: float = 5.0
    burst_limit: int = 10


@dataclass
class SyncResult:
    connector_id: str
    sync_job_id: str
    status: ConnectorStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_discovered: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    total_duplicates: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Optional[str] = None  # For resume capability

    @property
    def success_rate(self) -> float:
        if self.total_discovered == 0:
            return 1.0
        return self.total_processed / self.total_discovered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "sync_job_id": self.sync_job_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_discovered": self.total_discovered,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_skipped": self.total_skipped,
            "total_duplicates": self.total_duplicates,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "checkpoint": self.checkpoint,
        }


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def with_retry(max_retries: int = 3, backoff_base: float = 2.0, backoff_max: float = 300.0):
    """Decorator that wraps async functions with exponential backoff retry logic."""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt == max_retries:
                        logger.error(
                            "Max retries reached",
                            func=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        raise

                    delay = min(backoff_base ** attempt, backoff_max)
                    logger.warning(
                        "Retrying after error",
                        func=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Base connector
# ---------------------------------------------------------------------------

class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.

    Subclasses must implement:
        - discover()         → yield item identifiers
        - extract(item_id)   → fetch raw item from source
        - normalize(raw)     → convert to CanonicalDocument

    Everything else (auth refresh, retry, checkpointing, rate limiting,
    progress tracking, error recovery) is handled here.
    """

    SOURCE_TYPE: SourceType  # Must be set by each subclass

    def __init__(self, config: ConnectorConfig, redis: Redis, db: AsyncSession):
        self.config = config
        self.redis = redis
        self.db = db
        self.connector_id = config.connector_id
        self._status = ConnectorStatus.IDLE
        self._current_job_id: Optional[str] = None
        self._rate_limiter = asyncio.Semaphore(config.burst_limit)
        self._request_times: List[float] = []

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Perform or validate authentication.
        Returns True if auth is valid, False if re-auth needed.
        """
        ...

    @abstractmethod
    async def discover(self, checkpoint: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield item descriptors (id + metadata) from the source.
        Each yielded dict must contain at minimum: {"id": str, "type": str}
        checkpoint — if provided, resume from this position.
        """
        ...

    @abstractmethod
    async def extract(self, item_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch the full raw content of a single item from the source.
        Returns raw dict; normalization happens separately.
        """
        ...

    @abstractmethod
    async def normalize(self, raw_item: Dict[str, Any]) -> CanonicalDocument:
        """
        Transform source-specific raw data into our canonical format.
        After this, the item is indistinguishable from any other source.
        """
        ...

    @abstractmethod
    async def health_check(self) -> Tuple[bool, str]:
        """
        Check if this connector is healthy.
        Returns (is_healthy: bool, message: str)
        """
        ...

    # ------------------------------------------------------------------
    # Sync orchestration — the main engine
    # ------------------------------------------------------------------

    async def run_sync(
        self,
        mode: SyncMode = SyncMode.INCREMENTAL,
        job_id: Optional[str] = None,
    ) -> SyncResult:
        """
        Main sync entry point. Orchestrates discovery → extraction → normalization.
        Handles checkpointing, error recovery, and progress reporting.
        """
        job_id = job_id or str(uuid.uuid4())
        self._current_job_id = job_id
        self._status = ConnectorStatus.RUNNING

        started_at = datetime.now(timezone.utc)
        result = SyncResult(
            connector_id=self.connector_id,
            sync_job_id=job_id,
            status=ConnectorStatus.RUNNING,
            started_at=started_at,
        )

        logger.info(
            "Starting sync",
            connector_id=self.connector_id,
            job_id=job_id,
            mode=mode.value,
        )

        await self._publish_progress(job_id, 0, "starting", "Sync started")

        try:
            # Authenticate first
            auth_ok = await self.authenticate()
            if not auth_ok:
                raise RuntimeError("Authentication failed — re-authorization required")

            # Load checkpoint for resume mode
            checkpoint = None
            if mode in (SyncMode.INCREMENTAL, SyncMode.RESUME):
                checkpoint = await self._load_checkpoint(job_id if mode == SyncMode.RESUME else None)

            # Discovery phase — collect all item IDs
            items_to_process: List[Dict[str, Any]] = []
            async for item_desc in self.discover(checkpoint=checkpoint):
                items_to_process.append(item_desc)
                result.total_discovered += 1

                # Publish discovery progress every 100 items
                if result.total_discovered % 100 == 0:
                    await self._publish_progress(
                        job_id,
                        0,  # We don't know total yet
                        "discovering",
                        f"Discovered {result.total_discovered} items...",
                    )

            logger.info(
                "Discovery complete",
                connector_id=self.connector_id,
                total=result.total_discovered,
            )

            # Extraction + normalization phase
            total = len(items_to_process)
            for idx, item_desc in enumerate(items_to_process):
                try:
                    progress_pct = int((idx / total) * 100) if total > 0 else 0

                    # Rate limiting
                    await self._rate_limit()

                    # Extract raw data
                    raw = await self._extract_with_retry(item_desc)

                    # Normalize to canonical format
                    canonical = await self.normalize(raw)

                    # Store canonical document
                    await self._store_canonical(canonical)

                    result.total_processed += 1

                    # Save checkpoint every 50 items
                    if idx % 50 == 0:
                        await self._save_checkpoint(job_id, item_desc.get("id", str(idx)))
                        await self._publish_progress(
                            job_id,
                            progress_pct,
                            "processing",
                            f"Processed {result.total_processed}/{total} items ({progress_pct}%)",
                        )

                    # Log each item to sync log
                    await self._log_sync_item(job_id, item_desc["id"], "success")

                except Exception as e:
                    result.total_failed += 1
                    error_entry = {
                        "item_id": item_desc.get("id", "unknown"),
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    result.errors.append(error_entry)

                    logger.error(
                        "Item processing failed",
                        connector_id=self.connector_id,
                        item_id=item_desc.get("id"),
                        error=str(e),
                    )
                    await self._log_sync_item(job_id, item_desc.get("id", "unknown"), "failed", str(e))

            # Mark complete
            result.status = ConnectorStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)
            self._status = ConnectorStatus.COMPLETED

            await self._clear_checkpoint(job_id)
            await self._publish_progress(job_id, 100, "completed", f"Sync complete. {result.total_processed} items processed.")
            await self._save_sync_result(result)

            logger.info(
                "Sync completed",
                connector_id=self.connector_id,
                processed=result.total_processed,
                failed=result.total_failed,
                skipped=result.total_skipped,
            )

        except asyncio.CancelledError:
            result.status = ConnectorStatus.PAUSED
            self._status = ConnectorStatus.PAUSED
            await self._publish_progress(job_id, -1, "paused", "Sync paused — can be resumed")
            logger.warning("Sync cancelled/paused", connector_id=self.connector_id, job_id=job_id)
            raise

        except Exception as e:
            result.status = ConnectorStatus.FAILED
            result.completed_at = datetime.now(timezone.utc)
            self._status = ConnectorStatus.FAILED
            result.errors.append({
                "fatal": True,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self._publish_progress(job_id, -1, "failed", f"Sync failed: {e}")
            await self._save_sync_result(result)
            logger.error("Sync failed", connector_id=self.connector_id, error=str(e))
            raise

        return result

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """Simple token bucket rate limiter."""
        import time
        now = time.monotonic()
        # Remove timestamps older than 1 second
        self._request_times = [t for t in self._request_times if now - t < 1.0]
        if len(self._request_times) >= self.config.requests_per_second:
            sleep_time = 1.0 - (now - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self._request_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Retry extraction
    # ------------------------------------------------------------------

    async def _extract_with_retry(self, item_desc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract with exponential backoff retry."""
        max_retries = self.config.max_retries
        for attempt in range(max_retries + 1):
            try:
                return await self.extract(item_desc)
            except Exception as e:
                if attempt == max_retries:
                    raise
                delay = min(
                    self.config.retry_backoff_base ** attempt,
                    self.config.retry_backoff_max,
                )
                logger.warning(
                    "Extract failed, retrying",
                    item_id=item_desc.get("id"),
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Checkpointing (resume capability)
    # ------------------------------------------------------------------

    async def _save_checkpoint(self, job_id: str, last_processed_id: str) -> None:
        key = f"sync:checkpoint:{self.connector_id}:{job_id}"
        await self.redis.set(key, last_processed_id, ex=86400 * 7)

    async def _load_checkpoint(self, job_id: Optional[str] = None) -> Optional[str]:
        if job_id:
            key = f"sync:checkpoint:{self.connector_id}:{job_id}"
        else:
            # Load the most recent checkpoint for incremental sync
            key = f"sync:last_sync:{self.connector_id}"
        val = await self.redis.get(key)
        return val.decode() if val else None

    async def _clear_checkpoint(self, job_id: str) -> None:
        key = f"sync:checkpoint:{self.connector_id}:{job_id}"
        await self.redis.delete(key)
        # Save the completion time as the next incremental sync marker
        await self.redis.set(
            f"sync:last_sync:{self.connector_id}",
            datetime.now(timezone.utc).isoformat(),
            ex=86400 * 365,
        )

    # ------------------------------------------------------------------
    # Progress broadcasting
    # ------------------------------------------------------------------

    async def _publish_progress(
        self,
        job_id: str,
        percent: int,
        stage: str,
        message: str,
    ) -> None:
        """Publish progress updates to Redis pub/sub for real-time frontend."""
        payload = json.dumps({
            "connector_id": self.connector_id,
            "job_id": job_id,
            "percent": percent,
            "stage": stage,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await self.redis.publish(f"sync:progress:{self.connector_id}", payload)
        # Also write to a key for polling fallback
        await self.redis.set(
            f"sync:progress_state:{self.connector_id}",
            payload,
            ex=3600,
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def _store_canonical(self, doc: CanonicalDocument) -> None:
        """Queue canonical document for downstream processing."""
        from app.tasks.processing import process_canonical_document
        # Enqueue to Celery for async processing
        process_canonical_document.delay(doc.to_dict())

    async def _log_sync_item(
        self,
        job_id: str,
        item_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Write per-item sync log entry to Redis stream (durable log)."""
        await self.redis.xadd(
            f"sync:log:{job_id}",
            {
                "item_id": item_id,
                "status": status,
                "error": error or "",
                "connector_id": self.connector_id,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            maxlen=100000,
        )

    async def _save_sync_result(self, result: SyncResult) -> None:
        """Persist final sync result to Redis (and optionally PostgreSQL)."""
        await self.redis.set(
            f"sync:result:{result.sync_job_id}",
            json.dumps(result.to_dict()),
            ex=86400 * 30,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> ConnectorStatus:
        return self._status

    async def get_last_sync_time(self) -> Optional[str]:
        val = await self.redis.get(f"sync:last_sync:{self.connector_id}")
        return val.decode() if val else None
