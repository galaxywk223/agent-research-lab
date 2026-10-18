"""Scoped observational identities and redacted provider facts."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import uuid
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


def summary_objective_id(template: str, date: str, output_schema: dict) -> str:
    """Fingerprint the actual non-content prompt fields and output contract.

    Query, topic and task are absent from the summarization prompt. Web content
    is separately fingerprinted as summary_input_hash, never normalized here.
    """
    import unicodedata

    payload = {
        "version": 2,
        "template": unicodedata.normalize("NFC", template.replace("\r\n", "\n")),
        "date": unicodedata.normalize("NFC", date).strip(),
        "output_schema": output_schema,
    }
    return digest(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


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

    async def _journal(self, event_type: str, event: dict) -> None:
        import asyncio

        from agent_research_lab.incremental import append_event

        await asyncio.to_thread(append_event, self.run_id, self.research_id, event_type, event)

    def initialize_observations(self) -> None:
        """Initialize event storage without external counters."""
        self.result_events = []
        self.summary_events = []
        self.tool_events = []
        self.task_attempts = {}
        self._arrival_urls: set[str] = set()
        self._arrival_hashes: set[str] = set()
        self._arrival_domains: set[str] = set()
        self._task_arrival_urls: dict[str, set[str]] = {}
        self._task_arrival_hashes: dict[str, set[str]] = {}
        self._task_arrival_domains: dict[str, set[str]] = {}
        self._round_domain_counts: dict[tuple, int] = {}
        self._arrival_content_objectives: dict[tuple[str, str | None], str] = {}
        self._arrival_url_objectives: dict[tuple[str, str | None], str] = {}

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
            await self._journal("llm_started", event)
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
            await self._journal("llm_finished", event)

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
                "search_request_id": str(uuid.uuid4()),
            }
            self.search_request_events.append(event)
            self.search["search_provider_requests"] += 1
            if event["task_id"]:
                self.task(event["task_id"]).search_provider_requests += 1
                if event["round_index"]:
                    self._round_for(
                        event["task_id"], event["round_index"]
                    ).search_provider_requests += 1
            await self._journal("search_request_started", event)
            return event

    async def result_observation(
        self,
        result: dict,
        query: dict,
        rank: int,
        representative: str | None,
        summary_objective_id: str | None = None,
    ) -> dict:
        """Capture raw result facts without altering selection or model inputs."""
        import math
        import uuid

        from agent_research_lab.evidence import normalize_url
        from agent_research_lab.metrics import extract_domain

        async with self._lock:
            raw = result.get("raw_content")
            score = result.get("score")
            score = (
                score
                if isinstance(score, int | float) and math.isfinite(score)
                else None
            )
            task_id = self.event_scope().get("task_id")
            task_urls = self._task_arrival_urls.setdefault(task_id or "", set())
            task_hashes = self._task_arrival_hashes.setdefault(task_id or "", set())
            task_domains = self._task_arrival_domains.setdefault(task_id or "", set())

            normalized = result.get("normalized_url") or normalize_url(
                result.get("url", "")
            )
            raw_content = result.get("raw_content")
            content_hash = (
                result.get("raw_content_hash") or digest(raw_content)
                if isinstance(raw_content, str)
                else result.get("raw_content_hash")
            )
            domain = extract_domain(normalized)
            round_key = (
                task_id or "",
                self.event_scope().get("attempt_id"),
                self.event_scope().get("round_index"),
                domain or "",
            )
            domain_count = self._round_domain_counts.get(round_key, 0)
            content_previous_objective = (
                self._arrival_content_objectives.get(
                    (content_hash, summary_objective_id)
                )
                if content_hash and summary_objective_id
                else None
            )
            same_content_different_objective = bool(content_hash) and any(
                digest == content_hash and objective != summary_objective_id
                for (digest, objective) in self._arrival_content_objectives
            )
            same_url_previous_objective = (
                self._arrival_url_objectives.get((normalized, summary_objective_id))
                if normalized and summary_objective_id
                else None
            )
            same_url_different_objective = bool(normalized) and any(
                url == normalized and objective != summary_objective_id
                for (url, objective) in self._arrival_url_objectives
            )
            event = {
                **self.event_scope(),
                "result_id": str(uuid.uuid4()),
                "query_id": query.get("query_id"),
                "rank": rank,
                "result_rank": rank,
                "url": result.get("url"),
                "original_url": result.get("url"),
                "normalized_url": normalize_url(result["url"]),
                "title": result.get("title"),
                "score": score,
                "provider_score": score,
                "domain": extract_domain(result["url"]),
                "raw_content_available": bool(raw),
                "raw_content_char_count": len(raw) if raw is not None else None,
                "raw_content_hash": digest(raw) if isinstance(raw, str) else None,
                "snippet_char_count": len(result.get("content") or ""),
                "snippet_hash": digest(result.get("content") or ""),
                "duplicate_of_result_id": representative,
                "selected_after_raw_url_dedup": representative is None,
                "summary_objective_id": summary_objective_id,
                "summary_objective_version": 2 if summary_objective_id else None,
                "same_content_same_objective_before": content_previous_objective
                is not None,
                "same_content_different_objective_before": same_content_different_objective,
                "same_url_same_objective_before": same_url_previous_objective
                is not None,
                "same_url_different_objective_before": same_url_different_objective,
                "domain_repeat_only_before": bool(domain in self._arrival_domains)
                and normalized not in self._arrival_urls
                and content_hash not in self._arrival_hashes
                and not same_content_different_objective
                and not same_url_different_objective,
                "url_seen_before_in_task": normalized in task_urls
                if normalized
                else False,
                "url_seen_before_in_research": normalized in self._arrival_urls
                if normalized
                else False,
                "domain_seen_before_in_task": domain in task_domains
                if domain
                else False,
                "domain_seen_before_in_research": domain in self._arrival_domains
                if domain
                else False,
                "content_hash_seen_before_in_task": content_hash in task_hashes
                if content_hash
                else False,
                "content_hash_seen_before_in_research": content_hash
                in self._arrival_hashes
                if content_hash
                else False,
                "domain_seen_count_in_task_round": domain_count,
            }
            self.result_events.append(event)
            if normalized:
                task_urls.add(normalized)
                self._arrival_urls.add(normalized)
            if content_hash:
                task_hashes.add(content_hash)
                self._arrival_hashes.add(content_hash)
                if summary_objective_id:
                    self._arrival_content_objectives[
                        (content_hash, summary_objective_id)
                    ] = event["result_id"]
            if normalized and summary_objective_id:
                self._arrival_url_objectives[(normalized, summary_objective_id)] = (
                    event["result_id"]
                )
            if domain:
                task_domains.add(domain)
                self._arrival_domains.add(domain)
                self._round_domain_counts[round_key] = domain_count + 1
            from agent_research_lab.shadow_gate import shadow_trace

            event["shadow_gate_mode"] = "observe_only"
            event["shadow_decisions"] = shadow_trace(event)["policies"]
            await self._journal("result_arrived", event)
        return event

    async def finish_search_request(self, event: dict) -> None:
        """Persist the completed provider request snapshot."""
        await self._journal("search_request_finished", event)

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
            await self._journal("summary_started", event)
            if input_text is None:
                for result_row in self.result_events:
                    if result_row.get("result_id") == event.get("result_id"):
                        result_row["summarization_status"] = "skipped_no_raw_content"
                        result_row["summary_call_id"] = event["summary_call_id"]
        return event

    async def finish_summary(self, text: str, outcome: str) -> None:
        """Observe summary output or unchanged raw-content fallback."""
        async with self._lock:
            for event in self.summary_events:
                if event["summary_call_id"] == scope.get().get("summary_call_id"):
                    event.update(
                        outcome=outcome,
                        processing_outcome={
                            "summarized": "success",
                            "timeout_raw_fallback": "timeout",
                            "error_raw_fallback": "error",
                        }.get(outcome, outcome),
                        fallback_disposition=outcome.endswith("raw_fallback"),
                        summary_output_hash=digest(text),
                        output_hash=digest(text),
                        output_char_count=len(text),
                    )
                    for result in self.result_events:
                        if result.get("result_id") == event.get("result_id"):
                            result["summarization_status"] = outcome
                            result["summary_call_id"] = event.get("summary_call_id")
                    if outcome == "timeout_raw_fallback":
                        for llm in self.llm_events:
                            if (
                                llm["logical_llm_call_id"]
                                == event["logical_llm_call_id"]
                            ):
                                llm["status"] = "timeout"
                    await self._journal("summary_finished", event)

    async def observe_summary_objective(self, objective_id: str) -> None:
        """Bind the objective computed from the actual prompt's date and schema."""
        async with self._lock:
            summary_id = scope.get().get("summary_call_id")
            for summary in self.summary_events:
                if summary["summary_call_id"] == summary_id:
                    summary["summary_objective_id"] = objective_id
                    for result in self.result_events:
                        if result["result_id"] == summary["result_id"]:
                            result["summary_objective_id"] = objective_id
                    await self._journal("summary_objective", summary)
