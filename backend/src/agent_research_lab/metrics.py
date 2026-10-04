"""Non-invasive runtime instrumentation for the frozen baseline strategy."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler

from agent_research_lab.evidence import content_hash, normalize_url
from agent_research_lab.observability import ObservationMixin, scope

logger = logging.getLogger(__name__)
_current: contextvars.ContextVar[ResearchMetrics | None] = contextvars.ContextVar(
    "agentlab_metrics", default=None
)
_task: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentlab_metrics_task", default=None
)
_round: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "agentlab_metrics_round", default=None
)
_attempt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentlab_metrics_attempt", default=None
)
_collectors: dict[str, ResearchMetrics] = {}


def utc_now_iso() -> str:
    """Return a timezone-aware timestamp for record metadata."""
    return datetime.now(timezone.utc).isoformat()


def extract_domain(url: str) -> str | None:
    """Extract a normalized hostname for descriptive metrics only."""
    normalized = normalize_url(url) or url
    host = urlsplit(normalized).hostname
    if not host:
        return None
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def extract_usage(value: Any) -> dict[str, int] | None:
    """Read provider usage metadata without estimating missing token counts."""
    candidates = []
    for obj in (
        value,
        getattr(value, "usage_metadata", None),
        getattr(value, "response_metadata", None),
        getattr(value, "llm_output", None),
    ):
        if isinstance(obj, dict):
            candidates.append(obj)
            for key in ("usage", "token_usage", "usage_metadata"):
                if isinstance(obj.get(key), dict):
                    candidates.append(obj[key])
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for candidate in candidates:
        parsed: dict[str, int] = {}
        for target, keys in aliases.items():
            for key in keys:
                if isinstance(candidate.get(key), int | float):
                    parsed[target] = int(candidate[key])
                    break
        if parsed:
            if (
                "total_tokens" not in parsed
                and {"prompt_tokens", "completion_tokens"} <= parsed.keys()
            ):
                parsed["total_tokens"] = (
                    parsed["prompt_tokens"] + parsed["completion_tokens"]
                )
            return parsed
    return None


class _UsageObserver(BaseCallbackHandler):
    """Capture model output metadata beneath structured-output wrappers."""

    def __init__(self) -> None:
        self.usages: list[dict[str, int]] = []
        self.model_facts = {}

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Capture only non-secret resolved model parameters."""
        from agent_research_lab.observability import safe_endpoint

        params = kwargs.get("invocation_params") or {}
        self.model_facts["requested_model"] = params.get("model_name") or params.get(
            "model"
        )
        self.model_facts["base_url"] = safe_endpoint(
            params.get("openai_api_base") or os.getenv("OPENAI_BASE_URL")
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Keep only usage actually returned by the provider."""
        del kwargs
        usage = extract_usage(response)
        if usage is None:
            for generation in getattr(response, "generations", []):
                for item in generation:
                    usage = extract_usage(getattr(item, "message", None))
                    metadata = getattr(
                        getattr(item, "message", None), "response_metadata", {}
                    )
                    self.model_facts["returned_model"] = metadata.get(
                        "model_name"
                    ) or metadata.get("model")
                    if usage is not None:
                        break
        if usage is not None:
            self.usages.append(usage)


class ResearchMetrics(ObservationMixin):
    """Per-research collector safe for asyncio concurrency."""

    schema_version = 2

    def __init__(
        self,
        research_id: str,
        run_id: str,
        thread_id: str,
        config_snapshot: dict[str, Any] | None = None,
    ):
        """Create a collector for one research Run."""
        self.research_id, self.run_id, self.thread_id = research_id, run_id, thread_id
        self.started_at = utc_now_iso()
        self._started = time.perf_counter()
        self._lock = asyncio.Lock()
        self.config_snapshot = config_snapshot or {}
        self.llm_calls = Counter()
        self.llm_events: list[dict[str, Any]] = []
        self.search_request_events: list[dict[str, Any]] = []
        self.provider_llm_attempts: int | None = None
        self.usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self.token_usage_complete = True
        self.search = Counter()
        self.tasks: dict[str, TaskMetrics] = {}
        self.rounds: list[ResearcherRoundMetrics] = []
        self._seen_urls: set[str] = set()
        self._seen_hashes: set[str] = set()
        self._seen_domains: set[str] = set()
        self._finished = False
        self._terminal_snapshot = None
        self._finalize_lock = asyncio.Lock()
        self.status = "running"
        self.failure = {}
        self.question_id = None
        self.persisted_association_count = None
        self.initialize_observations()
        self.final_report: str | None = None
        self.source_urls: list[str] = []
        self.incomplete = False
        self.usage_source = (
            "simulated" if os.getenv("AGENTLAB_PROVIDER_MODE") == "fake" else "provider"
        )
        self.persisted_evidence_count: int | None = None
        self.persisted_hash_duplicates: int | None = None

    async def llm(
        self,
        role: str,
        response: Any,
        duration_ms: float | None = None,
        observed_usage: list[dict[str, int]] | None = None,
    ) -> None:
        """Record one logical invocation and provider-reported usage."""
        event = await self.start_llm(role)
        await self.finish_llm(event, response, observed_usage or [])
        event["latency_ms"] = duration_ms

    async def provider_request(
        self, query: str, latency_ms: float, result_count: int | None
    ) -> None:
        """Record one actual Tavily request's monotonic duration."""
        async with self._lock:
            self.search_request_events.append(
                {
                    **self.event_scope(),
                    "query": query,
                    "latency_ms": round(latency_ms, 3),
                    "result_count": result_count,
                }
            )

    def task(self, task_id: str, topic: str = "") -> TaskMetrics:
        """Get a task-local metrics bucket."""
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskMetrics(task_id, topic)
        return self.tasks[task_id]

    async def search_event(
        self,
        queries: list[str],
        raw_count: int,
        deduped_count: int,
        urls: list[str] | None = None,
        hashes: list[str] | None = None,
    ) -> None:
        """Record one search tool call and its request/result counts."""
        async with self._lock:
            scoped_tool = scope.get().get("search_tool_call_id")
            task_id = _task.get()
            task = self.task(task_id) if task_id else None
            round_index = _round.get()
            round_metrics = (
                self._round_for(task_id, round_index)
                if task_id and round_index
                else None
            )
            if not scoped_tool:
                self.search["search_tool_calls"] += 1
                self.search["search_query_count"] += len(queries)
                self.search["search_provider_requests"] += len(queries)
                if task:
                    task.search_calls += 1
                    task.search_query_count += len(queries)
                    task.search_provider_requests += len(queries)
                if round_metrics:
                    round_metrics.search_tool_calls += 1
                    round_metrics.search_query_count += len(queries)
                    round_metrics.search_provider_requests += len(queries)
                    round_metrics.queries.extend(queries)
            self.search["raw_result_count"] += raw_count
            self.search["deduped_result_count"] += deduped_count
            if task:
                task.raw_search_result_count += raw_count
            if round_metrics:
                round_metrics.raw_result_count += raw_count
                round_metrics.deduped_result_count += deduped_count
            for url in urls or []:
                normalized = normalize_url(url)
                if not normalized:
                    continue
                self.search["duplicate_url_count"] += normalized in self._seen_urls
                self._seen_urls.add(normalized)
                domain = extract_domain(normalized)
                if domain:
                    self._seen_domains.add(domain)
                if round_metrics:
                    round_metrics.returned_urls.append(url)
                    round_metrics.normalized_urls.append(normalized)
                    if domain:
                        round_metrics.domains.append(domain)
                if task:
                    task.duplicate_url_count += normalized in task.urls
                    task.urls.add(normalized)
                    if domain:
                        task.domains.add(domain)
            if round_metrics:
                round_metrics.content_hashes.extend(hashes or [])

    async def evidence_event(self, observations: list[dict[str, Any]]) -> None:
        """Record hashes of Evidence content submitted to persistence."""
        async with self._lock:
            for observation in observations:
                digest = content_hash(
                    observation.get("content"), observation.get("snippet")
                )
                if digest:
                    self.search["content_hash_duplicate_count"] += (
                        digest in self._seen_hashes
                    )
                    self._seen_hashes.add(digest)

    def _round_for(self, task_id: str, round_index: int) -> ResearcherRoundMetrics:
        for round_metrics in self.rounds:
            if (
                round_metrics.task_id == task_id
                and round_metrics.round_index == round_index
                and getattr(round_metrics, "attempt_id", None) == _attempt.get()
            ):
                return round_metrics
        round_metrics = ResearcherRoundMetrics(
            task_id, round_index, [], [], [], [], [], 0, 0, 0, 0, 0, 0.0
        )
        round_metrics.attempt_id = _attempt.get()
        self.rounds.append(round_metrics)
        return round_metrics

    async def add_round(
        self,
        task_id: str,
        round_index: int,
        queries: list[str],
        returned_urls: list[str],
        content_hashes: list[str],
        latency_ms: float,
    ) -> None:
        """Record novelty within one task's independent round sequence."""
        async with self._lock:
            self._add_round(
                task_id, round_index, queries, returned_urls, content_hashes, latency_ms
            )

    def _add_round(
        self,
        task_id: str,
        round_index: int,
        queries: list[str],
        returned_urls: list[str],
        content_hashes: list[str],
        latency_ms: float,
    ) -> None:
        bucket = self._round_for(task_id, round_index)
        snapshot_urls = bucket.returned_urls or returned_urls
        normalized = [u for u in (normalize_url(x) for x in snapshot_urls) if u]
        domains = [d for d in (extract_domain(u) for u in normalized) if d]
        task = self.task(task_id)
        before = {
            u
            for item in self.rounds
            if item.task_id == task_id
            and item.round_index < round_index
            and getattr(item, "attempt_id", None) == _attempt.get()
            for u in item.normalized_urls
        }
        before_domains = {
            d
            for item in self.rounds
            if item.task_id == task_id
            and item.round_index < round_index
            and getattr(item, "attempt_id", None) == _attempt.get()
            for d in item.domains
        }
        task.urls.update(normalized)
        task.domains.update(domains)
        task.hashes.update(h for h in content_hashes if h)
        task.rounds += 1
        round_metrics = self._round_for(task_id, round_index)
        if not round_metrics.queries:
            round_metrics.queries = queries
        if not round_metrics.returned_urls:
            round_metrics.returned_urls = returned_urls
            round_metrics.normalized_urls = normalized
            round_metrics.domains = domains
        if content_hashes:
            round_metrics.content_hashes = content_hashes
        round_metrics.new_unique_urls = len(set(normalized) - before)
        round_metrics.repeated_urls = (
            len(set(normalized)) - round_metrics.new_unique_urls
        )
        round_metrics.duplicate_ratio = (
            round_metrics.repeated_urls / len(set(normalized)) if normalized else None
        )
        round_metrics.returned_unique_urls = len(set(normalized))
        round_metrics.domains_seen_this_round = len(set(domains))
        round_metrics.novelty_ratio = (
            round_metrics.new_unique_urls / len(set(normalized)) if normalized else None
        )
        round_metrics.new_domains = len(set(domains) - before_domains)
        round_metrics.domain_novelty_ratio = (
            round_metrics.new_domains / len(set(domains)) if domains else None
        )
        round_metrics.tool_text_extracted_urls = list(dict.fromkeys(returned_urls))
        round_metrics.cumulative_unique_urls = len(before | set(normalized))
        round_metrics.cumulative_unique_domains = len(before_domains | set(domains))
        round_metrics.round_latency_ms = latency_ms

    async def finalize(
        self,
        final_report: str | None,
        source_urls: list[str],
        session_id: str | None = None,
        *,
        status: str | None = None,
        failure: dict | None = None,
    ) -> None:
        """Freeze one terminal snapshot and retry persistence without changing it."""
        async with self._finalize_lock:
            if self._finished:
                return
            if self._terminal_snapshot is None:
                self.final_report = final_report
                self.source_urls = list(dict.fromkeys(source_urls))
                self.status = status or (
                    "incomplete" if self.incomplete else "completed"
                )
                self.failure = failure or {}
                if self.status == "failed":
                    self.incomplete = True
                elapsed = round((time.perf_counter() - self._started) * 1000, 3)
                finished_at = utc_now_iso()
                try:
                    if session_id:
                        from sqlalchemy import func, select

                        from agent_research_lab.database import database_session
                        from agent_research_lab.models import (
                            Evidence,
                            task_attempt_evidence,
                        )

                        async with database_session() as database:
                            hashes = (
                                await database.scalars(
                                    select(Evidence.content_hash).where(
                                        Evidence.session_id
                                        == uuid.UUID(str(session_id))
                                    )
                                )
                            ).all()
                            self.persisted_association_count = await database.scalar(
                                select(func.count())
                                .select_from(task_attempt_evidence)
                                .join(Evidence)
                                .where(
                                    Evidence.session_id == uuid.UUID(str(session_id))
                                )
                            )
                        self.persisted_evidence_count = len(hashes)
                        self.persisted_hash_duplicates = sum(
                            count - 1
                            for count in Counter(
                                value for value in hashes if value
                            ).values()
                        )
                except Exception as error:
                    from agent_research_lab.observability import error_facts

                    self.failure["evidence_counts_error"] = error_facts(error)
                async with self._lock:
                    record = self.to_record()
                    record["finished_at"] = finished_at
                    record["metrics"]["total_latency_ms"] = elapsed
                    self._terminal_snapshot = copy.deepcopy(record)
            tmp = None
            try:
                path = Path(os.getenv("AGENTLAB_METRICS_DIR", "eval/results"))
                path.mkdir(parents=True, exist_ok=True)
                target = path / f"{self.run_id}.jsonl"
                fd, tmp = tempfile.mkstemp(
                    prefix=".metrics-", suffix=".jsonl", dir=path
                )
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(
                        json.dumps(self._terminal_snapshot, ensure_ascii=False) + "\n"
                    )
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
                self._finished = True
                _collectors.pop(self.run_id, None)
            except Exception as error:
                from agent_research_lab.observability import error_facts

                logger.warning("metrics persistence failed: %s", error_facts(error))
            finally:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)

    def to_record(self) -> dict[str, Any]:
        """Create a machine-readable metrics record without report body."""
        if self._terminal_snapshot is not None:
            return copy.deepcopy(self._terminal_snapshot)
        llm_by_id = {
            event.get("logical_llm_call_id"): event for event in self.llm_events
        }
        serialized_summaries = []
        for summary in self.summary_events:
            row = dict(summary)
            linked = llm_by_id.get(row.get("logical_llm_call_id"), {})
            row["usage"] = linked.get("usage")
            row["usage_complete"] = bool(linked.get("usage")) and all(
                linked.get("usage", {}).get(key) is not None
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            row["latency_ms"] = linked.get("latency_ms")
            serialized_summaries.append(row)
        funnel = {
            "tavily_raw_results": self.search.get("raw_result_count", 0),
            "tavily_deduped_results": self.search.get("deduped_result_count", 0),
            "summarized_results": sum(
                row.get("outcome")
                in {
                    "running",
                    "summarized",
                    "timeout_raw_fallback",
                    "error_raw_fallback",
                }
                for row in serialized_summaries
            ),
            "persisted_source_urls": self.persisted_evidence_count,
        }
        return {
            "schema_version": 2,
            "status": self.status,
            **self.failure,
            "strategy": "baseline_fixed",
            "research_id": self.research_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "started_at": self.started_at,
            "metrics": {
                "total_latency_ms": round(
                    (time.perf_counter() - self._started) * 1000, 3
                ),
                "logical_llm_calls": sum(self.llm_calls.values()),
                "supervisor_llm_calls": self.llm_calls["supervisor"],
                "researcher_llm_calls": self.llm_calls["researcher"],
                "compression_llm_calls": self.llm_calls["compression"],
                "final_report_llm_calls": self.llm_calls["final_report"],
                "clarification_llm_calls": self.llm_calls["clarification"],
                "research_brief_llm_calls": self.llm_calls["research_brief"],
                "summarization_llm_calls": self.llm_calls["summarization"],
                "provider_llm_attempts": self.provider_llm_attempts,
                "provider_attempts_supported": False,
                **{
                    key: self.search[key]
                    for key in (
                        "search_tool_calls",
                        "search_query_count",
                        "search_provider_requests",
                        "raw_result_count",
                        "deduped_result_count",
                        "duplicate_url_count",
                    )
                },
                **(
                    self.usage
                    if self.token_usage_complete
                    else dict.fromkeys(self.usage)
                ),
                "token_usage_complete": self.token_usage_complete,
                "researcher_count": len(self.tasks),
                "task_count": len(self.tasks),
                "total_researcher_rounds": len(self.rounds),
                "final_report_char_count": len(self.final_report or ""),
                "final_report_sha256": hashlib.sha256(
                    self.final_report.encode("utf-8")
                ).hexdigest()
                if self.final_report
                else None,
                "source_url_count": len(self.source_urls),
                "unique_url_count": len(self._seen_urls),
                "unique_domain_count": len(self._seen_domains),
                "supervisor_iterations": self.llm_calls["supervisor"],
                "evidence_count": self.persisted_evidence_count,
                "persisted_evidence_row_count": self.persisted_evidence_count,
                "task_attempt_evidence_association_count": self.persisted_association_count,
                "search_result_unique_url_count": len(self._seen_urls),
                "tavily_raw_result_count": self.search.get("raw_result_count", 0),
                "tavily_deduped_result_count": self.search.get(
                    "deduped_result_count", 0
                ),
                "summarized_result_count": sum(
                    row.get("outcome")
                    in {"summarized", "timeout_raw_fallback", "error_raw_fallback"}
                    for row in self.summary_events
                ),
                "persisted_source_url_count": self.persisted_evidence_count,
                "content_hash_duplicate_count": self.persisted_hash_duplicates,
            },
            "question_id": self.question_id,
            "finished_at": utc_now_iso(),
            "complete": not self.incomplete,
            "usage_source": self.usage_source,
            "config": self.config_snapshot,
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "rounds": [round_.to_dict() for round_ in self.rounds],
            "llm_events": [
                {key: value for key, value in event.items() if not key.startswith("_")}
                for event in self.llm_events
            ],
            "result_events": self.result_events,
            "summary_events": serialized_summaries,
            "search_tool_events": self.tool_events,
            "metric_semantics": {
                "evidence_count": "Session-level normalized URLs extracted from tool text; not Tavily result count",
                "content_hash_duplicate_count": "database aggregate research-note hashes; not webpage duplicates",
                "novelty": "per Task/Attempt unique search-result URLs and hostnames",
            },
            "result_processing_funnel": funnel,
            "search_request_events": self.search_request_events,
        }


class TaskMetrics:
    """Counters and evidence sets scoped to one business Task."""

    def __init__(self, task_id: str, topic: str):
        """Initialize Task metrics at claim time."""
        self.task_id, self.research_topic, self.started = (
            task_id,
            topic,
            time.perf_counter(),
        )
        self.urls: set[str] = set()
        self.domains: set[str] = set()
        self.hashes: set[str] = set()
        self.rounds = 0
        self.llm_calls = 0
        self.search_calls = 0
        self.search_query_count = 0
        self.search_provider_requests = 0
        self.raw_search_result_count = 0
        self.duplicate_url_count = 0
        self.usage: dict[str, int | None] = dict.fromkeys(
            ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        self.token_usage_complete = True
        self.status = "running"
        self.attempt_no: int | None = None
        self.latency_ms: float | None = None

    def finish(self, status: str) -> None:
        """Freeze Task duration when its Attempt commits."""
        self.status = status
        self.latency_ms = round((time.perf_counter() - self.started) * 1000, 3)

    def to_dict(self):
        """Return serializable Task metrics."""
        return {
            "task_id": self.task_id,
            "research_topic": self.research_topic,
            "latency_ms": self.latency_ms,
            "researcher_rounds": self.rounds,
            "logical_llm_calls": self.llm_calls,
            "provider_llm_attempts": None,
            "search_tool_calls": self.search_calls,
            "search_query_count": self.search_query_count,
            "search_provider_requests": self.search_provider_requests,
            "raw_search_result_count": self.raw_search_result_count,
            **(self.usage if self.token_usage_complete else dict.fromkeys(self.usage)),
            "token_usage_complete": self.token_usage_complete,
            "evidence_count": len(self.urls),
            "unique_url_count": len(self.urls),
            "unique_domain_count": len(self.domains),
            "duplicate_url_count": self.duplicate_url_count,
            "result": self.status,
            "attempt_no": self.attempt_no,
        }


class ResearcherRoundMetrics:
    """One researcher model/tools cycle."""

    def __init__(
        self,
        task_id,
        round_index,
        queries,
        returned_urls,
        normalized_urls,
        domains,
        content_hashes,
        new_unique_urls,
        repeated_urls,
        new_domains,
        cumulative_unique_urls,
        cumulative_unique_domains,
        round_latency_ms,
    ):
        """Initialize the round snapshot."""
        self.task_id = task_id
        self.round_index = round_index
        self.queries = queries
        self.search_tool_calls = 0
        self.search_query_count = 0
        self.search_provider_requests = 0
        self.raw_result_count = 0
        self.deduped_result_count = 0
        self.returned_urls = returned_urls
        self.normalized_urls = normalized_urls
        self.domains = domains
        self.content_hashes = content_hashes
        self.new_unique_urls = new_unique_urls
        self.repeated_urls = repeated_urls
        self.duplicate_ratio = (
            repeated_urls / len(set(normalized_urls)) if normalized_urls else 0.0
        )
        self.new_domains = new_domains
        self.cumulative_unique_urls = cumulative_unique_urls
        self.cumulative_unique_domains = cumulative_unique_domains
        self.round_latency_ms = round_latency_ms

    def to_dict(self):
        """Return serializable Round metrics."""
        return self.__dict__.copy()


def begin_research(
    research_id: str,
    run_id: str,
    thread_id: str,
    config_snapshot: dict[str, Any] | None = None,
) -> ResearchMetrics:
    """Register a collector for one Run."""
    collector = ResearchMetrics(research_id, run_id, thread_id, config_snapshot)
    _current.set(collector)
    _collectors[run_id] = collector
    return collector


def current_metrics() -> ResearchMetrics | None:
    """Return the current Task context's collector."""
    return _current.get()


def metrics_for(run_id: str | None) -> ResearchMetrics | None:
    """Find the current Run's collector without sharing counters between Runs."""
    return _collectors.get(str(run_id)) if run_id else None


def activate_run(run_id: str | None) -> ResearchMetrics | None:
    """Activate the collector in a LangGraph node's async context."""
    metrics = metrics_for(run_id)
    _current.set(metrics)
    return metrics


def bind_task(task_id: str) -> contextvars.Token:
    """Bind a Task for nested asynchronous model and tool calls."""
    return _task.set(task_id)


def reset_task(token: contextvars.Token) -> None:
    """Restore the previous Task context."""
    _task.reset(token)


def bind_round(round_index: int) -> contextvars.Token:
    """Bind a Researcher round for nested search calls."""
    return _round.set(round_index)


def reset_round(token: contextvars.Token) -> None:
    """Restore the previous round context."""
    _round.reset(token)


async def instrument_llm(invoke, role: str, *args, **kwargs):
    """Observe one logical LLM call without altering its result."""
    observer = _UsageObserver()
    original_config = kwargs.get("config")
    if isinstance(original_config, dict):
        invocation_config = dict(original_config)
        invocation_config["callbacks"] = list(
            original_config.get("callbacks") or []
        ) + [observer]
    else:
        invocation_config = {"callbacks": [observer]}
    kwargs["config"] = invocation_config
    collector = current_metrics()
    event = None
    if collector:
        try:
            event = await collector.start_llm(role)
        except Exception:
            logger.warning("LLM observation start failed")
    role_token = scope.set({**scope.get(), "llm_role": role})
    try:
        response = await invoke(*args, **kwargs)
    except BaseException as error:
        if collector and event:
            try:
                await collector.finish_llm(
                    event, None, observer.usages, error, observer.model_facts
                )
            except Exception:
                logger.warning("LLM observation finish failed")
        scope.reset(role_token)
        raise
    scope.reset(role_token)
    if collector and event:
        try:
            await collector.finish_llm(
                event, response, observer.usages, model_facts=observer.model_facts
            )
        except Exception:
            logger.warning("LLM observation finish failed")
    return response


def config_snapshot(config: Any) -> dict[str, Any]:
    """Capture resolved non-secret baseline configuration."""
    try:
        from agent_research_lab.configuration import Configuration

        c = Configuration.from_runnable_config(config)
        return {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in c.model_dump().items()
            if k not in {"mcp_config", "mcp_prompt"}
        }
    except Exception:
        return {}
