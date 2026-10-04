"""Explicit, journaled synthetic Judge acceptance without benchmark execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from agent_research_lab.metrics import extract_usage
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from eval.common import ROOT, atomic_json, emit, read_json, sha256, utc_now
from eval.judge_synthetic import SCHEMAS, acceptance_hashes, fixtures, payload, snapshot

MODELS = ("openai:deepseek-v4-pro-0813", "openai:glm-5.3")
METHODS = ("json_schema", "function_calling", "json_mode")
BUDGET = 8192


def scrub(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Remove credential values and credential-bearing fields recursively."""
    if isinstance(value, dict):
        return {
            key: scrub(item, secrets)
            for key, item in value.items()
            if not any(
                term in key.lower()
                for term in (
                    "authorization",
                    "api_key",
                    "access_token",
                    "secret",
                    "cookie",
                )
            )
        }
    if isinstance(value, list):
        return [scrub(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def wire_method(body: dict) -> str | None:
    """Identify the actual structured-output mechanism sent over HTTP."""
    mode = (body.get("response_format") or {}).get("type")
    if mode == "json_schema":
        return "json_schema"
    if mode == "json_object":
        return "json_mode"
    if body.get("tools"):
        return "function_calling"
    return None


def validate_output(kind: str, data: Any) -> dict:
    """Apply unchanged schemas and evaluator's coverage-index invariant."""
    result = {
        "validation_success": False,
        "parsed": None,
        "validation_errors": [],
        "missing_fields": False,
        "enum_violation": False,
        "range_violation": False,
    }
    try:
        value = SCHEMAS[kind].model_validate(data)
        if kind == "coverage":
            indices = sorted(item.requirement_index for item in value.requirements)
            if indices != [0, 1, 2, 3]:
                result["missing_fields"] = True
                result["validation_errors"] = [
                    {
                        "type": "coverage_indices",
                        "msg": "Every requirement must be scored exactly once",
                    }
                ]
                return result
        result.update(validation_success=True, parsed=value.model_dump())
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        result["validation_errors"] = errors
        result["missing_fields"] = any(error["type"] == "missing" for error in errors)
        result["enum_violation"] = any(
            error["type"] in {"literal_error", "enum"} for error in errors
        )
        result["range_violation"] = any(
            error["type"] in {"greater_than_equal", "less_than_equal", "int_type"}
            for error in errors
        )
    return result


class WireObserver:
    """Capture safe request fields and response bodies; never capture headers."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        """Initialize storage local to one logical invocation."""
        self.secrets = secrets
        self.requests: list[dict] = []
        self.responses: list[dict] = []

    async def request(self, request: httpx.Request) -> None:
        """Record the submitted model parameters and method, excluding messages."""
        body = json.loads(request.content)
        self.requests.append(
            scrub(
                {
                    "path": request.url.path,
                    "model": body.get("model"),
                    "temperature": body.get("temperature"),
                    "temperature_present": "temperature" in body,
                    "max_tokens": body.get("max_tokens"),
                    "max_completion_tokens": body.get("max_completion_tokens"),
                    "enable_thinking": body.get("enable_thinking"),
                    "reasoning_effort": body.get("reasoning_effort"),
                    "method": wire_method(body),
                    "response_format": body.get("response_format"),
                    "tool_schemas": body.get("tools"),
                    "tool_choice": body.get("tool_choice"),
                    "messages_sha256": sha256(
                        json.dumps(
                            body.get("messages"), sort_keys=True, ensure_ascii=False
                        ).encode()
                    ),
                },
                self.secrets,
            )
        )

    async def response(self, response: httpx.Response) -> None:
        """Record sanitized provider outputs and errors before SDK parsing."""
        await response.aread()
        try:
            body = response.json()
        except ValueError:
            body = {
                "error": {
                    "type": "non_json_response",
                    "message": "Response body omitted",
                }
            }
        self.responses.append(
            scrub({"http_status": response.status_code, "body": body}, self.secrets)
        )


def response_facts(observer: WireObserver) -> dict:
    """Extract observable usage, reasoning and finish state without guessing."""
    body = observer.responses[-1]["body"] if observer.responses else {}
    choices = body.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = body.get("usage")
    details = (usage or {}).get("completion_tokens_details") or {}
    return {
        "api_success": any(
            200 <= item["http_status"] < 300 for item in observer.responses
        ),
        "http_status": observer.responses[-1]["http_status"]
        if observer.responses
        else None,
        "returned_model": body.get("model"),
        "usage": extract_usage({"usage": usage}) if usage else None,
        "usage_metadata": usage,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "reasoning_present": bool(reasoning),
        "reasoning_char_count": len(str(reasoning)),
        "reasoning_sha256": sha256(str(reasoning).encode()) if reasoning else None,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "content_char_count": len(str(message.get("content") or "")),
        "temperature_effective": "unknown",
        "thinking_mode_confirmed": "unknown",
    }


async def invoke(
    model: str,
    profile: dict,
    kind: str | None,
    method: str | None,
    inputs: dict | str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Execute exactly one no-retry ChatOpenAI invocation and preserve failures."""
    keys = (os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"),)
    base_url = os.getenv("EVAL_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not keys[0] or not base_url:
        raise ValueError("Local OpenAI-compatible credential and base URL are required")
    observer = WireObserver(tuple(key for key in keys if key))
    record: dict[str, Any] = {
        "requested_model": model,
        "profile": profile,
        "kind": kind,
        "requested_method": method,
        "started_at": utc_now(),
        "input_sha256": sha256(
            json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()
        ),
        "parsed_success": False,
        "validation_success": False,
        "fallback": False,
        "raw_extra_content": False,
        "error": None,
        "usage_source": "provider" if transport is None else "simulated",
    }
    started = time.perf_counter()
    warning_messages = []
    async with httpx.AsyncClient(
        transport=transport,
        timeout=180,
        event_hooks={"request": [observer.request], "response": [observer.response]},
    ) as client:
        params = {
            "model": model.removeprefix("openai:"),
            "api_key": keys[0],
            "base_url": base_url,
            "max_retries": 0,
            "max_tokens": BUDGET,
            "temperature": profile.get("temperature"),
            "timeout": 180,
            "http_async_client": client,
            "use_responses_api": False,
        }
        if "enable_thinking" in profile:
            params["extra_body"] = {"enable_thinking": profile["enable_thinking"]}
        if "reasoning_effort" in profile:
            params["reasoning_effort"] = profile["reasoning_effort"]
        found = []
        try:
            with warnings.catch_warnings(record=True) as found:
                warnings.simplefilter("always")
                chat = ChatOpenAI(**params)
                if kind:
                    system = (ROOT / f"prompts/{kind}_judge_v1.txt").read_text(
                        encoding="utf-8"
                    )
                    runnable = chat.with_structured_output(
                        SCHEMAS[kind], method=method, include_raw=True
                    )
                    messages = [
                        SystemMessage(content=system),
                        HumanMessage(content=json.dumps(inputs, ensure_ascii=False)),
                    ]
                else:
                    runnable = chat
                    messages = [HumanMessage(content=str(inputs))]
                output = await runnable.ainvoke(messages)
            if kind:
                parsed = output.get("parsed")
                record["parsed_success"] = (
                    parsed is not None and output.get("parsing_error") is None
                )
                if parsed is not None:
                    record.update(validate_output(kind, parsed))
                else:
                    # Validate provider JSON independently to characterize malformed outputs even if parser rejected them.
                    record["parsing_error"] = type(output.get("parsing_error")).__name__
            else:
                record.update(
                    parsed_success=True,
                    validation_success=bool(output.content),
                    basic_content=output.content,
                )
        except Exception as exc:
            record["error"] = scrub(
                {"type": type(exc).__name__, "message": str(exc)[:2000]},
                observer.secrets,
            )
        warning_messages = [str(item.message) for item in found]
        record.update(response_facts(observer))
        if kind and not record.get("validation_success"):
            body = observer.responses[-1]["body"] if observer.responses else {}
            choices = body.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            try:
                calls = message.get("tool_calls") or []
                text = (
                    calls[0]["function"]["arguments"]
                    if calls
                    else message.get("content", "")
                )
                independent = validate_output(kind, json.loads(text))
                record.update(
                    {
                        key: independent[key]
                        for key in (
                            "validation_errors",
                            "missing_fields",
                            "enum_violation",
                            "range_violation",
                        )
                    }
                )
            except (ValueError, TypeError, KeyError):
                pass
        actual_methods = [item["method"] for item in observer.requests]
        record["fallback"] = bool(
            kind and any(actual != method for actual in actual_methods)
        )
        record["langchain_warnings"] = scrub(warning_messages, observer.secrets)
        if any(
            "falling back" in warning.lower() or "overriding to" in warning.lower()
            for warning in warning_messages
        ):
            record["fallback"] = True
        body = observer.responses[-1]["body"] if observer.responses else {}
        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content")
        if method == "function_calling":
            record["raw_extra_content"] = bool(content)
        elif kind and isinstance(content, str):
            try:
                json.loads(content)
            except ValueError:
                record["raw_extra_content"] = True
        record["truncated"] = record["finish_reason"] in {"length", "max_tokens"}
        record["success"] = bool(
            record["api_success"]
            and record["parsed_success"]
            and record["validation_success"]
            and not record["fallback"]
            and not record["truncated"]
        )
        record.update(
            wire_requests=observer.requests,
            wire_responses=observer.responses,
            http_requests=len(observer.requests),
            finished_at=utc_now(),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
    return scrub(record, observer.secrets)


def select_profile(records: dict[str, dict]) -> dict | None:
    """Prefer explicit disabled thinking when the observed response supports it."""
    disabled = records.get("thinking_off") or {}
    if (
        disabled.get("success")
        and not disabled.get("reasoning_present")
        and disabled.get("reasoning_tokens") in (None, 0)
    ):
        return {"temperature": 0, "enable_thinking": False}
    low = records.get("reasoning_low") or {}
    if low.get("success"):
        return {"temperature": 0, "reasoning_effort": "low"}
    if (records.get("basic_en") or {}).get("success"):
        return {"temperature": 0}
    if (records.get("basic_zh") or {}).get("success"):
        return {}
    return None


def select_method(rows: list[dict]) -> str | None:
    """Select only an explicitly tested primary method passing all three schemas."""
    for method in ("json_schema", "function_calling"):
        matches = [row for row in rows if row.get("requested_method") == method]
        if (
            len(matches) == 3
            and {row.get("kind") for row in matches} == set(SCHEMAS)
            and all(row.get("success") and not row.get("fallback") for row in matches)
        ):
            return method
    return None


class Acceptance:
    """Execute a fixed test schedule with atomic per-request journals."""

    def __init__(self, directory: Path) -> None:
        """Initialize an isolated acceptance directory."""
        self.directory = directory
        self.records: list[dict] = []

    async def call(
        self,
        model: str,
        name: str,
        phase: str,
        profile: dict,
        kind: str | None = None,
        method: str | None = None,
        case: str = "C",
        swapped: bool = False,
        long: bool = False,
        text: str = "",
    ) -> dict:
        """Reuse a completed acceptance invocation but never silently retry one."""
        slug = "deepseek" if model == MODELS[0] else "glm"
        path = self.directory / "calls" / f"{slug}-{name}.json"
        inputs = payload(kind, case, swapped, long) if kind else text
        identity = {
            "model": model,
            "phase": phase,
            "profile": profile,
            "kind": kind,
            "method": method,
            "case": case,
            "swapped": swapped,
            "long": long,
            "input_sha256": sha256(
                json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()
            ),
        }
        if path.exists():
            row = read_json(path)
            if row["identity"] != identity:
                raise ValueError(
                    "Acceptance invocation changed; create a new acceptance directory"
                )
            if row.get("status") == "starting":
                raise RuntimeError(
                    "Interrupted request has unknown outcome; do not silently resend"
                )
            record = row["record"]
        else:
            atomic_json(
                path,
                {"identity": identity, "status": "starting", "started_at": utc_now()},
            )
            record = await invoke(model, profile, kind, method, inputs)
            record.update(
                call_id=f"{slug}-{name}",
                phase=phase,
                case=case,
                swapped=swapped,
                long=long,
            )
            atomic_json(
                path, {"identity": identity, "status": "finished", "record": record}
            )
        self.records.append(record)
        emit(
            {
                "call": record["call_id"],
                "phase": phase,
                "success": record["success"],
                "http_status": record["http_status"],
                "latency_ms": record["latency_ms"],
                "usage": record["usage"],
            }
        )
        return record

    async def model(self, model: str) -> dict:
        """Run basic, parameter, matrix, repeat and length stages for one candidate."""
        basic = {}
        basic["basic_zh"] = await self.call(
            model,
            "basic-zh",
            "basic",
            {},
            text="虚构星桥方案每趟20箱、120元，只在白天使用。请用中文一句话复述这三个事实，不加入新事实。",
        )
        basic["basic_en"] = await self.call(
            model,
            "basic-en",
            "basic",
            {"temperature": 0},
            text="Fictional Starbridge carries 20 boxes per trip, costs 120 credits, and operates in daytime. Restate these facts in one English sentence without new claims.",
        )
        if not any(item["success"] for item in basic.values()):
            return {
                "model": model,
                "not_run": ["thinking", "matrix", "reliability", "length"],
                "reason": "Basic generation unavailable",
            }
        for name, config in (
            ("thinking_off", {"temperature": 0, "enable_thinking": False}),
            ("thinking_on", {"temperature": 0, "enable_thinking": True}),
            ("reasoning_low", {"temperature": 0, "reasoning_effort": "low"}),
        ):
            basic[name] = await self.call(
                model,
                name,
                "thinking",
                config,
                text="Compare fictional plans: Starbridge carries 20 boxes for 120 credits in daytime; Rainboat carries 12 boxes for 80 credits at night. Give one sentence about a tradeoff. There is no safety data.",
            )
        profile = select_profile(basic)
        if profile is None:
            return {
                "model": model,
                "not_run": ["matrix", "reliability", "length"],
                "reason": "No callable fixed profile",
            }
        matrix = []
        for method in METHODS:
            for kind in SCHEMAS:
                matrix.append(
                    await self.call(
                        model,
                        f"matrix-{method}-{kind}",
                        "matrix",
                        profile,
                        kind,
                        method,
                        case="obvious" if kind == "pairwise" else "C",
                    )
                )
        method = select_method(matrix)
        if method is None:
            return {
                "model": model,
                "profile": profile,
                "method": None,
                "not_run": ["reliability", "length"],
                "reason": "No primary structured method passed all schemas",
            }
        for kind in ("coverage", "quality"):
            for index in range(10):
                await self.call(
                    model,
                    f"repeat-{kind}-C-{index:02}",
                    "reliability",
                    profile,
                    kind,
                    method,
                )
        for swapped in (False, True):
            for index in range(10):
                await self.call(
                    model,
                    f"repeat-pair-obvious-{int(swapped)}-{index:02}",
                    "reliability",
                    profile,
                    "pairwise",
                    method,
                    "obvious",
                    swapped,
                )
            for index in range(4):
                await self.call(
                    model,
                    f"repeat-pair-near-{int(swapped)}-{index:02}",
                    "reliability",
                    profile,
                    "pairwise",
                    method,
                    "near",
                    swapped,
                )
        for case in ("A", "B"):
            for kind in ("coverage", "quality"):
                await self.call(
                    model,
                    f"sanity-{kind}-{case}",
                    "reliability",
                    profile,
                    kind,
                    method,
                    case,
                )
        for kind in ("coverage", "quality"):
            await self.call(
                model, f"long-{kind}", "length", profile, kind, method, "A", long=True
            )
        return {"model": model, "profile": profile, "method": method, "not_run": []}


async def run(directory: Path) -> None:
    """Execute only the approved synthetic acceptance schedule, at most 136 calls."""
    load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")
    initial = snapshot()
    hashes = acceptance_hashes()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest["acceptance_hashes"] != hashes
            or manifest["protected_hashes"] != initial
        ):
            raise ValueError("Acceptance fixtures or protected assets changed")
        if manifest.get("recommended_config_sha256"):
            if (
                sha256((directory / "recommended_config.json").read_bytes())
                != manifest["recommended_config_sha256"]
            ):
                raise ValueError("Frozen recommendation changed")
            emit(
                {
                    "summary_path": str(directory / "summary.json"),
                    "calls": manifest["observed_calls"],
                    "recommendation": read_json(directory / "recommended_config.json"),
                    "reused_completed_acceptance": True,
                }
            )
            return
    else:
        manifest = {
            "started_at": utc_now(),
            "models": list(MODELS),
            "planned_max_calls": 136,
            "max_tokens_per_request": BUDGET,
            "max_retries": 0,
            "acceptance_hashes": hashes,
            "protected_hashes": initial,
            "versions": {
                name: version(name)
                for name in (
                    "langchain",
                    "langchain-core",
                    "langchain-openai",
                    "openai",
                    "pydantic",
                )
            },
            "gateway_host": urlsplit(
                os.getenv("EVAL_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
            ).hostname,
        }
        atomic_json(manifest_path, manifest)
        atomic_json(directory / "synthetic_inputs.json", fixtures())
        for kind in SCHEMAS:
            atomic_json(
                directory / "inputs" / f"long-{kind}.json",
                payload(kind, "A", long=True),
            )
    acceptance = Acceptance(directory)
    selected = []
    try:
        for model in MODELS:
            selected.append(await acceptance.model(model))
            atomic_json(
                directory / "progress.json",
                {"models": selected, "calls": len(acceptance.records)},
            )
    finally:
        current = snapshot()
        if current != initial:
            raise RuntimeError("Protected files changed during acceptance")
    from eval.judge_acceptance_analysis import summarize_acceptance

    result = summarize_acceptance(acceptance.records, selected)
    result.update(
        acceptance_hashes=hashes,
        protected_files_unchanged=True,
        versions=manifest["versions"],
    )
    atomic_json(directory / "summary.json", result)
    atomic_json(directory / "recommended_config.json", result["recommendation"])
    manifest.update(
        finished_at=utc_now(),
        observed_calls=len(acceptance.records),
        protected_files_unchanged=True,
    )
    atomic_json(manifest_path, manifest)
    emit(
        {
            "summary_path": str(directory / "summary.json"),
            "calls": len(acceptance.records),
            "recommendation": result["recommendation"],
        }
    )


def main() -> None:
    """Require an explicit real-execution flag for paid synthetic requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-real", action="store_true", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/judge-acceptance-v1"
    )
    args = parser.parse_args()
    asyncio.run(run(args.output_dir))


if __name__ == "__main__":
    main()
