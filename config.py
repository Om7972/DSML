"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-2026")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-dev-secret-2026")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 86400))

    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "health_symptomsense")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

    RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "200 per day;50 per hour")
    RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "20 per minute")

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "1") == "1"
    APP_VERSION = os.getenv("APP_VERSION", "3.0.0")

    GUNICORN_WORKERS = int(os.getenv("GUNICORN_WORKERS", 4))
    GUNICORN_THREADS = int(os.getenv("GUNICORN_THREADS", 2))
    GUNICORN_TIMEOUT = int(os.getenv("GUNICORN_TIMEOUT", 120))
