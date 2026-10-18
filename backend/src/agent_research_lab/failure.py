"""Root invocation failure finalization without changing graph routing."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass

from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.errors import GraphBubbleUp

from agent_research_lab.observability import error_facts

_execution = contextvars.ContextVar("research_execution", default=None)
logger = logging.getLogger(__name__)


@dataclass
class Execution:
    """Keep business identity independent of the LangChain trace UUID."""

    trace_id: str
    question_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None


def register_business(run_id: str, session_id: str) -> None:
    """Bind the actual created Session before the first provider invocation."""
    frame = _execution.get()
    if frame:
        frame.run_id, frame.session_id = run_id, session_id


def question_id() -> str | None:
    """Return the root invocation's external question identity when supplied."""
    frame = _execution.get()
    return frame.question_id if frame else None


class FailureObserver(AsyncCallbackHandler):
    """Finalize only unhandled errors of the main compiled graph."""

    run_inline = True

    async def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        """Create a coroutine-local frame for the outermost observed invocation."""
        if parent_run_id is None:
            _execution.set(
                Execution(
                    str(run_id), (kwargs.get("metadata") or {}).get("question_id")
                )
            )

    async def on_chain_error(
        self, error: BaseException, *, run_id, parent_run_id=None, **kwargs
    ):
        """Persist a partial terminal record while preserving the original error."""
        frame = _execution.get()
        if not frame or frame.trace_id != str(run_id) or parent_run_id is not None:
            return
        if isinstance(error, GraphBubbleUp | asyncio.CancelledError):
            return
        from agent_research_lab.metrics import metrics_for
        from agent_research_lab.repository import ResearchRepository

        metrics = metrics_for(frame.run_id)
        facts = error_facts(error)
        matching = (
            [e for e in metrics.llm_events if e.get("_error_identity") == id(error)]
            if metrics
            else []
        )
        matching_search = (
            [e for e in metrics.search_request_events if e.get("_error_identity") == id(error)]
            if metrics
            else []
        )
        stage = matching[-1]["role"] if matching else "tavily_search" if matching_search else "graph"
        facts["failure_stage"] = stage
        facts["failure_origin"] = "provider" if matching or matching_search or facts.get("http_status") else "agent"
        try:
            if frame.run_id:
                repository = ResearchRepository()
                if matching and matching[-1].get("attempt_id"):
                    import uuid

                    await repository.fail_owned_attempt(
                        uuid.UUID(matching[-1]["attempt_id"]),
                        frame.run_id,
                        facts["failure_type"] + ": " + facts["failure_message"],
                    )
                await repository.finalize_failed_run(frame.run_id, facts)
        except Exception as cleanup_error:
            facts["failure_origin"] = "business_persistence"
            facts["database_finalization_error"] = error_facts(cleanup_error)
            logger.warning(
                "Failure database transition failed: %s",
                facts["database_finalization_error"],
            )
        if metrics:
            try:
                await metrics.finalize(
                    None, [], frame.session_id, status="failed", failure=facts
                )
            except Exception as cleanup_error:
                logger.warning(
                    "Failure snapshot failed: %s", error_facts(cleanup_error)
                )

    async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        """Release only the current root's invocation frame."""
        frame = _execution.get()
        if frame and frame.trace_id == str(run_id) and parent_run_id is None:
            _execution.set(None)
