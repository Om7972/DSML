#!/usr/bin/env python3
"""Production deployment helper for Health SymptomSense."""

import subprocess
import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "docker"

    if mode == "docker":
        print("Starting Health SymptomSense with Docker Compose...")
        subprocess.run(["docker", "compose", "up", "--build", "-d"], check=True)
        print("\nApp:     http://localhost:5000")
        print("Metrics: http://localhost:5000/metrics")
        print("Health:  http://localhost:5000/api/health")
    elif mode == "gunicorn":
        print("Starting with Gunicorn (production WSGI)...")
        subprocess.run([
            sys.executable, "-m", "gunicorn",
            "-c", "gunicorn.conf.py", "main:app",
        ], check=True)
    elif mode == "dev":
        print("Starting development server...")
        subprocess.run([sys.executable, "main.py"], check=True)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python deploy.py [docker|gunicorn|dev]")
        sys.exit(1)


if __name__ == "__main__":
    main()
