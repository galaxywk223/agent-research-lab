"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import time
import uuid
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import get_runtime
from langgraph.types import Command

from agent_research_lab.configuration import (
    Configuration,
)
from agent_research_lab.errors import InvalidResearchBriefError
from agent_research_lab.evidence import (
    MIN_RESEARCH_BRIEF_LENGTH,
    append_sources,
    extract_source_urls,
    is_valid_research_brief,
)
from agent_research_lab.failure import FailureObserver, question_id, register_business
from agent_research_lab.metrics import (
    _attempt,
    _round,
    activate_run,
    begin_research,
    bind_round,
    bind_task,
    config_snapshot,
    instrument_llm,
    metrics_for,
    reset_round,
    reset_task,
)
from agent_research_lab.models import SessionRunStatus, SessionStatus
from agent_research_lab.observability import bind as bind_observation
from agent_research_lab.observability import error_facts, redact, scope
from agent_research_lab.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from agent_research_lab.providers import create_chat_model
from agent_research_lab.repository import ResearchRepository
from agent_research_lab.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from agent_research_lab.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = create_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

repository = ResearchRepository()


def _execution_ids(state: AgentState | SupervisorState | ResearcherState, config: RunnableConfig) -> tuple[str, str]:
    """Resolve the current Thread and Run identifiers without forcing UUID types."""
    thread_id = str(config.get("configurable", {}).get("thread_id") or "standalone")
    runtime_run_id = None
    try:
        runtime_run_id = get_runtime().execution_info.run_id
    except (AttributeError, RuntimeError, KeyError):
        runtime_run_id = None
    run_id = runtime_run_id or config.get("run_id") or state.get("run_id") or str(uuid.uuid4())
    return thread_id, str(run_id)


def _research_context(state: AgentState, config: RunnableConfig) -> tuple[str, bool, bool]:
    """Resolve explicit new/continuation Research context at the graph boundary."""
    configurable = config.get("configurable", {})
    continuation = bool(configurable.get("continuation", state.get("continuation", False)))
    new_research = bool(configurable.get("new_research", state.get("new_research", False)))
    research_id = configurable.get("research_id") or state.get("research_id")
    if continuation:
        if not research_id:
            raise ValueError("Continuation requires research_id in checkpoint state or invocation context")
        return str(research_id), True, False
    if new_research:
        return str(uuid.uuid4()), False, True
    if research_id:
        if str(configurable.get("run_id", "")) == str(state.get("run_id", "")):
            return str(research_id), False, False
        raise ValueError("Research context is ambiguous; specify new_research=true or continuation=true")
    return str(uuid.uuid4()), False, True


async def _ensure_session(state: AgentState, config: RunnableConfig):
    """Create or reuse the business Session for the current LangGraph Run."""
    thread_id, run_id = _execution_ids(state, config)
    research_id, continuation, _ = _research_context(state, config)
    user_query = get_buffer_string(state.get("messages", []))
    session = await repository.create_research_session(
        research_id=research_id,
        run_id=run_id,
        thread_id=thread_id,
        user_query=user_query,
        continuation=continuation,
        checkpoint_id=config.get("configurable", {}).get("checkpoint_id"),
        config=dict(config.get("configurable", {})),
    )
    register_business(run_id, str(session.id))
    return thread_id, run_id, research_id, session


async def _finish_run(state: AgentState, *, run_status: SessionRunStatus, session_status: SessionStatus, final_report: str | None = None) -> None:
    """Finish the current Run while preserving historical session_runs."""
    run_id = state.get("run_id")
    if run_id:
        await repository.finish_session_run(run_id, status=run_status, session_status=session_status)
        if final_report is not None and state.get("session_id"):
            await repository.update_session(uuid.UUID(str(state["session_id"])), status=session_status, final_report=final_report)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    metrics_started = time.perf_counter()
    _, run_id, research_id, session = await _ensure_session(state, config)
    if metrics_for(run_id) is None:
        begin_research(research_id, run_id, _execution_ids(state, config)[0], config_snapshot(config))._started = metrics_started
    else:
        activate_run(run_id)
    if metrics_for(run_id):
        metrics_for(run_id).question_id = question_id()
    session_update = {"run_id": run_id, "research_id": research_id, "session_id": str(session.id)}

    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief", update=session_update)
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await instrument_llm(clarification_model.ainvoke, "clarification", [HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        metrics = metrics_for(run_id)
        if metrics:
            metrics.incomplete = True
            await metrics.finalize(None, [], str(state.get("session_id")) if state.get("session_id") else None)
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)], **session_update}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)], **session_update}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    activate_run(state.get("run_id"))
    if state.get("session_id"):
        _, _, _, session = await _ensure_session(state, config)
    else:
        _, _, _, session = await _ensure_session(state, config)

    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await instrument_llm(research_model.ainvoke, "research_brief", [HumanMessage(content=prompt_content)])

    if not is_valid_research_brief(response.research_brief):
        metrics = metrics_for(_execution_ids(state, config)[1])
        if metrics:
            metrics.incomplete = True
            await metrics.finalize(None, [], str(state.get("session_id")) if state.get("session_id") else None)
        await _finish_run({"run_id": _execution_ids(state, config)[1], "session_id": str(session.id)}, run_status=SessionRunStatus.FAILED, session_status=SessionStatus.FAILED)
        raise InvalidResearchBriefError(
            f"Research brief must contain at least {MIN_RESEARCH_BRIEF_LENGTH} characters."
        )
    await repository.update_session(session.id, status=SessionStatus.RUNNING, research_brief=response.research_brief)
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "run_id": _execution_ids(state, config)[1],
            "session_id": str(session.id),
            "research_id": session.research_id,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    activate_run(state.get("run_id"))
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await instrument_llm(research_model.ainvoke, "supervisor", supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def _run_research_task(
    *,
    state: SupervisorState,
    config: RunnableConfig,
    tool_call: dict,
    task_order: int,
) -> dict:
    """Run one idempotent research Task and persist its Attempt result."""
    arguments = dict(tool_call.get("args", {}))
    topic = str(arguments.get("research_topic", "")).strip()
    session_id = state.get("session_id")
    if not session_id:
        raise RuntimeError("Supervisor state is missing session_id")
    task = await repository.get_or_create_task(
        session_id=uuid.UUID(str(session_id)),
        topic=topic,
        task_order=task_order,
        arguments=arguments,
    )
    metrics = metrics_for(state.get("run_id"))
    task_metrics = metrics.task(str(task.id), topic) if metrics else None
    claim = await repository.create_attempt_locked(task.id, run_id=state.get("run_id"))
    if claim.completed_attempt:
        if task_metrics:
            task_metrics.attempt_no = claim.completed_attempt.attempt_no
            task_metrics.finish("completed")
        return await repository.get_attempt_result(claim.completed_attempt.id)
    if claim.already_running:
        return {
            "compressed_research": "Research task is already running; no duplicate provider call was started.",
            "raw_notes": [],
            "source_urls": [],
            "already_running": True,
        }
    if not claim.attempt:
        raise RuntimeError(f"Unable to acquire attempt for task {task.id}")

    attempt_id = str(claim.attempt.id)
    _attempt.set(attempt_id)
    if task_metrics:
        task_metrics.attempt_no = claim.attempt.attempt_no
    try:
        await repository.refresh_attempt_lease(claim.attempt.id)
        observation = await researcher_subgraph.ainvoke(
            {
                "researcher_messages": [HumanMessage(content=topic)],
                "research_topic": topic,
                "run_id": state.get("run_id"),
                "session_id": session_id,
                "task_id": str(task.id),
                "attempt_id": attempt_id,
            },
            config,
        )
        compressed = str(observation.get("compressed_research", ""))
        raw_notes = "\n".join(str(item) for item in observation.get("raw_notes", []))
        source_urls = list(dict.fromkeys(observation.get("source_urls", [])))
        if compressed.startswith("Error synthesizing research report:"):
            raise RuntimeError(compressed)
        evidence = [{"url": url, "content": raw_notes, "snippet": compressed} for url in source_urls]
        await repository.complete_attempt_with_evidence(
            attempt_id=claim.attempt.id,
            compressed_research=compressed,
            raw_notes=raw_notes,
            observations=evidence,
        )
        if task_metrics:
            task_metrics.finish("completed")
            try:
                await metrics.evidence_event(evidence)
            except Exception:
                pass
        return observation
    except Exception as error:
        await repository.fail_attempt(claim.attempt.id, redact(str(error)))
        if task_metrics:
            task_metrics.finish("failed")
        return {
            "compressed_research": f"Error executing research task: {error}",
            "raw_notes": [],
            "source_urls": [],
            "failed": True,
        }


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        # Limit concurrent research units to prevent resource exhaustion.
        allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
        research_tasks = [
            _run_research_task(state=state, config=config, tool_call=tool_call, task_order=index)
            for index, tool_call in enumerate(allowed_conduct_research_calls)
        ]
        tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)

        normalized_results = []
        for result in tool_results:
            if isinstance(result, Exception):
                normalized_results.append({"compressed_research": f"Error executing research task: {result}", "raw_notes": [], "source_urls": [], "failed": True})
            else:
                normalized_results.append(result)

        for observation, tool_call in zip(normalized_results, allowed_conduct_research_calls):
            all_tool_messages.append(ToolMessage(
                content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))

        for overflow_call in overflow_conduct_research_calls:
            all_tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_call["id"],
            ))

        raw_notes_concat = "\n".join("\n".join(observation.get("raw_notes", [])) for observation in normalized_results)
        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]
        source_urls = [url for observation in normalized_results for url in observation.get("source_urls", [])]
        if source_urls:
            update_payload["source_urls"] = list(dict.fromkeys(source_urls))
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    metrics = activate_run(state.get("run_id"))
    _attempt.set(str(state.get("attempt_id")) if state.get("attempt_id") else None)
    _round.set(state.get("tool_call_iterations", 0) + 1)
    task_token = bind_task(str(state.get("task_id"))) if metrics and state.get("task_id") else None
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    if state.get("attempt_id"):
        await repository.refresh_attempt_lease(uuid.UUID(str(state["attempt_id"])))
    started = time.perf_counter()
    response = await instrument_llm(research_model.ainvoke, "researcher", messages)
    if metrics and state.get("task_id"):
        metrics.task(str(state["task_id"])).round_start = started
    if task_token:
        reset_task(task_token)
    if state.get("attempt_id"):
        await repository.refresh_attempt_lease(uuid.UUID(str(state["attempt_id"])))
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    metrics = activate_run(state.get("run_id"))
    task_token = bind_task(str(state.get("task_id"))) if metrics and state.get("task_id") else None
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        if metrics and state.get("task_id"):
            bucket = metrics.task(str(state["task_id"]))
            await metrics.add_round(str(state["task_id"]), state.get("tool_call_iterations", 0), [], [], [], (time.perf_counter() - getattr(bucket, "round_start", time.perf_counter())) * 1000)
        if task_token:
            reset_task(task_token)
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    async def observed_tool(call):
        metrics = metrics_for(state.get("run_id"))
        parent_id = metrics.parent_llm_id(str(state.get("task_id")), state.get("tool_call_iterations", 0), call.get("id")) if metrics else None
        token = bind_observation(agent_tool_call_id=call.get("id"), parent_logical_llm_call_id=parent_id)
        try:
            return await execute_tool_safely(tools_by_name[call["name"]], call["args"], config)
        finally:
            scope.reset(token)
    tool_execution_tasks = [observed_tool(call) for call in tool_calls]
    if state.get("attempt_id"):
        await repository.refresh_attempt_lease(uuid.UUID(str(state["attempt_id"])))
    round_token = bind_round(state.get("tool_call_iterations", 0)) if metrics else None
    observations = await asyncio.gather(*tool_execution_tasks)
    if round_token:
        reset_round(round_token)
    if state.get("attempt_id"):
        await repository.refresh_attempt_lease(uuid.UUID(str(state["attempt_id"])))
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]

    search_observations = [
        str(observation)
        for observation, tool_call in zip(observations, tool_calls)
        if tool_call["name"] != "think_tool"
    ]
    source_urls = extract_source_urls(search_observations)
    if metrics and state.get("task_id"):
        bucket = metrics.task(str(state["task_id"]))
        queries = [
            str(query)
            for call in tool_calls
            if call["name"] in {"tavily_search", "web_search"}
            for query in (call.get("args", {}).get("queries") or [call.get("args", {}).get("query")])
            if query
        ]
        await metrics.add_round(str(state["task_id"]), state.get("tool_call_iterations", 0), queries, source_urls, [], (time.perf_counter() - getattr(bucket, "round_start", time.perf_counter())) * 1000)
    if task_token:
        reset_task(task_token)
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs, "source_urls": source_urls}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs, "source_urls": source_urls}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    metrics = activate_run(state.get("run_id"))
    task_token = bind_task(str(state.get("task_id"))) if metrics and state.get("task_id") else None
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            if state.get("attempt_id"):
                await repository.refresh_attempt_lease(uuid.UUID(str(state["attempt_id"])))
            response = await instrument_llm(synthesizer_model.ainvoke, "compression", messages)
            if task_token:
                reset_task(task_token)
            if state.get("attempt_id"):
                await repository.refresh_attempt_lease(uuid.UUID(str(state["attempt_id"])))
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                "source_urls": state.get("source_urls", []),
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
        "source_urls": state.get("source_urls", []),
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    metrics = activate_run(state.get("run_id"))
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    source_urls = list(dict.fromkeys(state.get("source_urls", [])))
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    if not is_valid_research_brief(state.get("research_brief")):
        if state.get("session_id"):
            await _finish_run(state, run_status=SessionRunStatus.FAILED, session_status=SessionStatus.FAILED)
        if metrics:
            metrics.incomplete = True
            await metrics.finalize(None, [], str(state.get("session_id")) if state.get("session_id") else None, status="failed", failure={"failure_stage": "final_report", "failure_type": "ResearchPrerequisiteError", "failure_message": "Research brief incomplete"})
        return {
            "final_report": "研究简报不完整，已停止生成未经验证的报告。",
            "messages": [AIMessage(content="研究简报不完整，无法生成报告。")],
            **cleared_state,
        }
    if not notes or not source_urls:
        if state.get("session_id"):
            await _finish_run(state, run_status=SessionRunStatus.FAILED, session_status=SessionStatus.FAILED)
        if metrics:
            metrics.incomplete = True
            await metrics.finalize(None, source_urls, str(state.get("session_id")) if state.get("session_id") else None, status="failed", failure={"failure_stage": "final_report", "failure_type": "ReportGenerationFailed", "failure_message": "No successful final report"})
        return {
            "final_report": "未获得足够的研究记录或真实来源，已停止生成未经验证的报告。",
            "messages": [AIMessage(content="缺少真实检索证据，无法生成报告。")],
            **cleared_state,
        }
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await instrument_llm(configurable_model.with_config(writer_model_config).ainvoke, "final_report", [
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            if state.get("session_id"):
                report_text = append_sources(str(final_report.content), source_urls)
                await _finish_run(state, run_status=SessionRunStatus.COMPLETED, session_status=SessionStatus.COMPLETED, final_report=report_text)
            else:
                report_text = append_sources(str(final_report.content), source_urls)
            if metrics:
                await metrics.finalize(report_text, source_urls, str(state.get("session_id")) if state.get("session_id") else None)
            return {
                "final_report": report_text,
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        await _finish_run(state, run_status=SessionRunStatus.FAILED, session_status=SessionStatus.FAILED)
                        if metrics:
                            metrics.incomplete = True
                            await metrics.finalize(None, source_urls, str(state.get("session_id")) if state.get("session_id") else None, status="failed", failure={"failure_stage": "final_report", "failure_type": "ReportGenerationFailed", "failure_message": "No successful final report"})
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                if state.get("session_id"):
                    await _finish_run(state, run_status=SessionRunStatus.FAILED, session_status=SessionStatus.FAILED)
                if metrics:
                    metrics.incomplete = True
                    await metrics.finalize(None, source_urls, str(state.get("session_id")) if state.get("session_id") else None, status="failed", failure={"failure_stage": "final_report", **error_facts(e)})
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    if state.get("session_id"):
        await _finish_run(state, run_status=SessionRunStatus.FAILED, session_status=SessionStatus.FAILED)
    if metrics:
        metrics.incomplete = True
        await metrics.finalize(None, source_urls, str(state.get("session_id")) if state.get("session_id") else None, status="failed", failure={"failure_stage": "final_report", "failure_type": "ReportGenerationFailed", "failure_message": "No successful final report"})
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile().with_config({"callbacks": [FailureObserver()]})

# Stable public graph name used by the local workbench and frontend.
deep_research = deep_researcher
