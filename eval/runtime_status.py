"""Classify benchmark API health separately from Research outcomes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx


def api_runtime_status(
    health_url: str | None = None, record_path: str | Path | None = None
) -> dict:
    """Return process/health facts without inferring graph recoverability."""
    url = health_url or os.getenv("EVAL_API_HEALTH_URL", "http://127.0.0.1:2035/ok")
    path = Path(record_path or os.getenv(
        "EVAL_API_PROCESS_RECORD", ".runtime/dev-v2-api/process.json"
    ))
    record_error = None
    try:
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError) as error:
        record = {}
        record_error = type(error).__name__
    try:
        response = httpx.get(url, timeout=1.0)
        healthy = response.is_success
        status_code = response.status_code
    except httpx.HTTPError:
        healthy = False
        status_code = None
    process_state = record.get("state", "unknown")
    if healthy:
        state = "RUNNING"
    elif process_state == "exited":
        state = "API_PROCESS_EXITED"
    else:
        state = "API_UNREACHABLE"
    return {
        "state": state,
        "health_success": healthy,
        "health_status_code": status_code,
        "last_health_success_at": record.get("last_health_success_at"),
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": record.get("exit_code"),
        "stdout_log": record.get("stdout_log"),
        "stderr_log": record.get("stderr_log"),
        "api_pid": record.get("api_pid"),
        "graph_execution_recoverable": False,
        "process_record_error": record_error,
    }

