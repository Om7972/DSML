"""Redis-backed cache with in-memory fallback."""

import json
import logging
import time

logger = logging.getLogger(__name__)

_redis_client = None
_memory_cache = {}


def init_cache(redis_url=None):
    """Initialize Redis connection; falls back to in-memory cache."""
    global _redis_client
    if not redis_url:
        logger.info("No Redis URL configured; using in-memory cache")
        return False
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
        client.ping()
        _redis_client = client
        logger.info("Redis cache connected")
        return True
    except Exception as exc:
        logger.warning("Redis unavailable (%s); using in-memory cache", exc)
        _redis_client = None
        return False


def is_redis_available():
    return _redis_client is not None


def cache_get(key):
    if _redis_client:
        try:
            value = _redis_client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None
    entry = _memory_cache.get(key)
    if entry and entry["expires_at"] > time.time():
        return entry["value"]
    _memory_cache.pop(key, None)
    return None


def cache_set(key, value, ttl=3600):
    if _redis_client:
        try:
            _redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception:
            pass
    _memory_cache[key] = {"value": value, "expires_at": time.time() + ttl}
    return True


def cache_delete(key):
    if _redis_client:
        try:
            _redis_client.delete(key)
        except Exception:
            pass
    _memory_cache.pop(key, None)
