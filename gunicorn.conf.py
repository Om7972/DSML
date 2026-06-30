"""Gunicorn configuration for production deployment."""

import multiprocessing
import os

bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5000')}"
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
threads = int(os.getenv("GUNICORN_THREADS", 2))
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
worker_class = "gthread"
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = True
max_requests = 1000
max_requests_jitter = 50
