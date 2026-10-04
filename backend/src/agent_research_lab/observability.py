"""Scoped observational identities and redacted provider facts."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

scope: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "observation_scope", default={}
)


def bind(**fields: Any) -> contextvars.Token:
    """Extend only the current coroutine's observation context."""
    return scope.set({**scope.get(), **fields})


def digest(text: str) -> str:
    """Hash exact UTF-8 content without estimating tokens."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact(text: str) -> str:
    """Redact configured credentials and common credential spellings."""
    for name, value in os.environ.items():
        if value and any(
            part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"(?i)(?:bearer\s+|sk-)[\w.\-]+", "[REDACTED]", text)


def safe_endpoint(value: str | None) -> str | None:
    """Keep endpoint host/path while omitting credentials and query parameters."""
    if not value:
        return None
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def error_facts(error: BaseException) -> dict:
    """Extract only safe facts available on the actual exception."""
    request = getattr(error, "request", None)
    model = None
    if request is not None:
        try:
            model = json.loads(request.content).get("model")
        except (ValueError, AttributeError, TypeError):
            pass
    body = getattr(error, "body", None)
    body = body.get("error", body) if isinstance(body, dict) else {}
    message = redact(str(body.get("message") or error))
    return {
        "failure_type": f"{type(error).__module__}.{type(error).__name__}",
        "failure_message": message[:4096],
        "failure_message_truncated": len(message) > 4096,
        "http_status": getattr(error, "status_code", None),
        "provider_code": body.get("code") or body.get("type"),
        "requested_model": model,
        "endpoint": safe_endpoint(str(request.url)) if request is not None else None,
    }


class ObservationMixin:
    """Keep event association inside one Research collector."""

    def initialize_observations(self) -> None:
        """Initialize event storage without external counters."""
        self.result_events = []
        self.summary_events = []
        self.tool_events = []
        self.task_attempts = {}

    def event_scope(self) -> dict:
        """Return a copy of strictly bound identities."""
        from agent_research_lab.metrics import _attempt, _round, _task

        return {
            "research_id": self.research_id,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "task_id": _task.get(),
            "attempt_id": _attempt.get(),
            "round_index": _round.get(),
            **scope.get(),
        }

    async def start_llm(self, role: str) -> dict:
        """Count a logical invocation before awaiting the provider."""
        import time
        import uuid

        async with self._lock:
            event = {
                **self.event_scope(),
                "logical_llm_call_id": str(uuid.uuid4()),
                "role": role,
                "status": "running",
                "usage": None,
                "latency_ms": None,
                "_started": time.perf_counter(),
            }
            self.llm_calls[role] += 1
            if event["task_id"]:
                self.task(event["task_id"]).llm_calls += 1
            self.llm_events.append(event)
            for summary in self.summary_events:
                if summary["summary_call_id"] == event.get("summary_call_id"):
                    summary["logical_llm_call_id"] = event["logical_llm_call_id"]
            return event

    async def finish_llm(
        self,
        event: dict,
        response: Any,
        usages: list,
        error: BaseException | None = None,
        model_facts: dict | None = None,
    ) -> None:
        """Finish success, failure or cancellation without losing observed usage."""
        import asyncio
        import time

        from agent_research_lab.metrics import extract_usage

        async with self._lock:
            usage = extract_usage(response)
            if usages:
                usage = {
                    key: sum(row[key] for row in usages)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if all(key in row for row in usages)
                }
            event.update(
                status="cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "error"
                if error
                else "success",
                latency_ms=round(
                    (time.perf_counter() - event.pop("_started")) * 1000, 3
                ),
                usage=usage,
                **(model_facts or {}),
            )
            if error:
                event["_error_identity"] = id(error)
                event.update(error_facts(error))
            if response is not None:
                event["generated_tool_call_ids"] = [
                    call.get("id") for call in getattr(response, "tool_calls", [])
                ]
            task = self.task(event["task_id"]) if event["task_id"] else None
            if usage is None or len(usage) < 3:
                self.token_usage_complete = False
                if task:
                    task.token_usage_complete = False
            else:
                for key, value in usage.items():
                    self.usage[key] = (self.usage[key] or 0) + value
                    if task:
                        task.usage[key] = (task.usage[key] or 0) + value

    def parent_llm_id(
        self, task_id: str, round_index: int, tool_id: str | None
    ) -> str | None:
        """Associate tool execution by explicit task/round and returned tool ID."""
        matches = [
            event
            for event in self.llm_events
            if event["task_id"] == task_id
            and event["round_index"] == round_index
            and tool_id in event.get("generated_tool_call_ids", [])
        ]
        return matches[-1]["logical_llm_call_id"] if matches else None

    async def start_search_tool(self, queries: list[str]) -> dict:
        """Register each query occurrence and the actual tool invocation."""
        import uuid

        async with self._lock:
            event = {
                **self.event_scope(),
                "search_tool_call_id": str(uuid.uuid4()),
                "queries": [
                    {
                        "query_id": str(uuid.uuid4()),
                        "query": query,
                        "query_index": index,
                    }
                    for index, query in enumerate(queries)
                ],
            }
            self.tool_events.append(event)
            self.search["search_tool_calls"] += 1
            self.search["search_query_count"] += len(queries)
            if event["task_id"]:
                task = self.task(event["task_id"])
                task.search_calls += 1
                task.search_query_count += len(queries)
                if event["round_index"]:
                    row = self._round_for(event["task_id"], event["round_index"])
                    row.search_tool_calls += 1
                    row.search_query_count += len(queries)
                    row.queries.extend(queries)
            return event

    async def start_search_request(self, query: dict) -> dict:
        """Count an SDK request before awaiting a possibly failing search."""
        async with self._lock:
            event = {
                **self.event_scope(),
                **query,
                "status": "running",
                "latency_ms": None,
                "result_count": None,
                "provider_network_attempts": None,
            }
            self.search_request_events.append(event)
            self.search["search_provider_requests"] += 1
            if event["task_id"]:
                self.task(event["task_id"]).search_provider_requests += 1
                if event["round_index"]:
                    self._round_for(
                        event["task_id"], event["round_index"]
                    ).search_provider_requests += 1
            return event

    async def result_observation(
        self, result: dict, query: dict, rank: int, representative: str | None
    ) -> dict:
        """Capture raw result facts without altering selection or model inputs."""
        import math
        import uuid

        from agent_research_lab.evidence import normalize_url
        from agent_research_lab.metrics import extract_domain

        raw = result.get("raw_content")
        score = result.get("score")
        score = (
            score if isinstance(score, int | float) and math.isfinite(score) else None
        )
        event = {
            **self.event_scope(),
            "result_id": str(uuid.uuid4()),
            "query_id": query.get("query_id"),
            "rank": rank,
            "url": result.get("url"),
            "normalized_url": normalize_url(result["url"]),
            "title": result.get("title"),
            "score": score,
            "domain": extract_domain(result["url"]),
            "raw_content_available": bool(raw),
            "raw_content_char_count": len(raw) if raw is not None else None,
            "raw_content_hash": digest(raw) if isinstance(raw, str) else None,
            "snippet_char_count": len(result.get("content") or ""),
            "duplicate_of_result_id": representative,
            "selected_after_raw_url_dedup": representative is None,
        }
        async with self._lock:
            self.result_events.append(event)
        return event

    async def start_summary(self, result: dict, input_text: str | None) -> dict:
        """Record a selected result's summarization or no-raw-content branch."""
        import uuid

        event = {
            **result,
            "summary_call_id": str(uuid.uuid4()),
            "logical_llm_call_id": None,
            "summary_input_char_count": len(input_text)
            if input_text is not None
            else None,
            "summary_input_hash": digest(input_text)
            if input_text is not None
            else None,
            "outcome": "running"
            if input_text is not None
            else "skipped_no_raw_content",
        }
        async with self._lock:
            self.summary_events.append(event)
        return event

    async def finish_summary(self, text: str, outcome: str) -> None:
        """Observe summary output or unchanged raw-content fallback."""
        async with self._lock:
            for event in self.summary_events:
                if event["summary_call_id"] == scope.get().get("summary_call_id"):
                    event.update(
                        outcome=outcome,
                        output_hash=digest(text),
                        output_char_count=len(text),
                    )
                    if outcome == "timeout_raw_fallback":
                        for llm in self.llm_events:
                            if (
                                llm["logical_llm_call_id"]
                                == event["logical_llm_call_id"]
                            ):
                                llm["status"] = "timeout"
