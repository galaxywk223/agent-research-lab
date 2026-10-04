"""Provider factories used by production and deterministic API tests."""

from __future__ import annotations

import os
import uuid
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import PrivateAttr

from agent_research_lab.metrics import current_metrics


class FakeChatModel(BaseChatModel):
    """Deterministic model that exercises graph tool and structured-output paths."""

    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "agentlab-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeChatModel:
        """Return a model copy with the supplied tool definitions."""
        clone = self.model_copy(deep=True)
        clone._bound_tools = list(tools)
        return clone

    @staticmethod
    def _tool_name(tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name", "web_search"))
        return str(getattr(tool, "name", getattr(tool, "__name__", "tool")))

    @staticmethod
    def _last_human(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return "fake research topic"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        from agent_research_lab.observability import scope
        if os.getenv("AGENTLAB_FAKE_FAIL_ROLE") and os.getenv("AGENTLAB_FAKE_FAIL_ROLE") == scope.get().get("llm_role"):
            raise RuntimeError("Synthetic provider failure")
        usage = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
        names = [self._tool_name(tool) for tool in self._bound_tools]
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        structured = {"ClarifyWithUser", "ResearchQuestion"}.intersection(names)
        if "ClarifyWithUser" in structured:
            args = {"need_clarification": False, "question": "", "verification": "开始执行研究。"}
            call_name = "ClarifyWithUser"
        elif "ResearchQuestion" in structured:
            args = {"research_brief": "This is a deterministic fake research brief with enough detail to exercise the complete persistence and recovery workflow."}
            call_name = "ResearchQuestion"
        elif "ConductResearch" in names:
            if any(message.name == "ConductResearch" for message in tool_messages):
                args = {}
                call_name = "ResearchComplete"
            else:
                args = {"research_topic": self._last_human(messages)}
                call_name = "ConductResearch"
        elif "web_search" in names or "tavily_search" in names:
            if tool_messages:
                args = {}
                call_name = "ResearchComplete"
            else:
                args = {"query": self._last_human(messages)}
                call_name = next(name for name in names if name in {"web_search", "tavily_search"})
        else:
            content = "Fake report with verified source evidence: https://example.com/fake-source"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, usage_metadata=usage, response_metadata={"usage_source": "simulated"}))])
        message = AIMessage(
            content="",
            usage_metadata=usage,
            response_metadata={"usage_source": "simulated"},
            tool_calls=[{"name": call_name, "args": args, "id": f"fake-{uuid.uuid4().hex}"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def provider_mode() -> str:
    """Return the configured model/search provider mode."""
    return os.getenv("AGENTLAB_PROVIDER_MODE", "real").strip().lower()


def create_chat_model(**kwargs: Any) -> BaseChatModel:
    """Create a deterministic fake model or the configured real model."""
    if provider_mode() == "fake":
        return FakeChatModel()
    from langchain.chat_models import init_chat_model

    return init_chat_model(**kwargs)


def fake_search_tool() -> StructuredTool:
    """Return a deterministic search tool for API E2E tests."""
    async def search(query: str) -> str:
        metrics = current_metrics()
        if metrics:
            try:
                await metrics.search_event(
                    [query], 1, 1, ["https://example.com/fake-source"]
                )
            except Exception:
                pass
        return f"Fake search result for {query}: https://example.com/fake-source"

    return StructuredTool.from_function(
        coroutine=search,
        name="web_search",
        description="Deterministic fake web search for tests.",
    )
