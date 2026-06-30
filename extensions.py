"""Flask extensions: rate limiter, metrics, logging."""

import logging
import sys

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import Config

metrics = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_limiter(app):
    storage_uri = "memory://"
    if Config.REDIS_URL:
        try:
            import redis
            redis.from_url(Config.REDIS_URL, socket_connect_timeout=2).ping()
            storage_uri = Config.REDIS_URL
        except Exception:
            pass
    return Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[Config.RATE_LIMIT_DEFAULT],
        storage_uri=storage_uri,
    )


def init_extensions(app):
    global metrics

    setup_logging()
    limiter = create_limiter(app)
    app.extensions["limiter"] = limiter

    if Config.ENABLE_METRICS:
        try:
            from prometheus_flask_exporter import PrometheusMetrics
            metrics = PrometheusMetrics(app, path="/metrics")
            metrics.info("app_info", "Health SymptomSense application info", version=Config.APP_VERSION)
        except Exception as exc:
            logging.getLogger(__name__).warning("Prometheus metrics disabled: %s", exc)

    return limiter
