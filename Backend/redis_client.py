"""
Redis async client — used for caching entity lookups, dedup keys,
and pipeline coordination.
"""

from typing import Optional, Any
import json
import redis.asyncio as aioredis

from app.core.config import settings


_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def cache_set(key: str, value: Any, ttl: int = 3600) -> None:
    r = await get_redis()
    await r.setex(key, ttl, json.dumps(value))


async def cache_get(key: str) -> Optional[Any]:
    r = await get_redis()
    data = await r.get(key)
    if data:
        return json.loads(data)
    return None


async def cache_delete(key: str) -> None:
    r = await get_redis()
    await r.delete(key)


async def cache_exists(key: str) -> bool:
    r = await get_redis()
    return bool(await r.exists(key))


# ── Distributed lock helpers ──────────────────────────────────────────────────

async def acquire_lock(lock_key: str, ttl: int = 30) -> bool:
    """Try to acquire a distributed lock. Returns True if acquired."""
    r = await get_redis()
    return await r.set(lock_key, "1", nx=True, ex=ttl)


async def release_lock(lock_key: str) -> None:
    r = await get_redis()
    await r.delete(lock_key)


# ── Dedup set helpers ─────────────────────────────────────────────────────────

async def add_to_processed_set(set_key: str, item_id: str) -> bool:
    """Return True if item was NOT already in the set (first time seen)."""
    r = await get_redis()
    result = await r.sadd(set_key, item_id)
    return bool(result)


async def is_processed(set_key: str, item_id: str) -> bool:
    r = await get_redis()
    return bool(await r.sismember(set_key, item_id))
