"""Fake-transport checks for synthetic Judge acceptance diagnostics."""

from __future__ import annotations

import json

import httpx
import pytest

from eval.evaluate import DIMENSIONS
from eval.judge_acceptance import (
    MODELS,
    invoke,
    scrub,
    select_method,
    validate_output,
    wire_method,
)
from eval.judge_acceptance_analysis import logical_winner, rates
from eval.judge_synthetic import payload


def valid_quality() -> dict:
    """Return one schema-valid quality fixture, not a real judgment."""
    return {
        **dict.fromkeys(DIMENSIONS, 3),
        "reasons": dict.fromkeys(DIMENSIONS, "synthetic reason"),
    }


def mock_transport(
    data: dict,
    method: str = "json_schema",
    extra_text: bool = False,
    usage: bool = True,
) -> httpx.MockTransport:
    """Return a successful provider response using the actual wire mechanism."""

    async def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert wire_method(body) == method
        message: dict = {"role": "assistant", "content": json.dumps(data)}
        if method == "function_calling":
            message["content"] = "additional text" if extra_text else None
            message["tool_calls"] = [
                {
                    "id": "call-synthetic",
                    "type": "function",
                    "function": {
                        "name": body["tools"][0]["function"]["name"],
                        "arguments": json.dumps(data),
                    },
                }
            ]
        response = {
            "id": "synthetic-response",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                    if method == "function_calling"
                    else "stop",
                }
            ],
        }
        if usage:
            response["usage"] = {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handle)


@pytest.fixture(autouse=True)
def local_config(monkeypatch):
    """Use only dummy credentials and an in-memory HTTP transport."""
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "synthetic-test-secret")
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "https://synthetic.invalid/v1")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["json_schema", "function_calling", "json_mode"])
async def test_actual_langchain_structured_paths(method):
    """Exercise the installed ChatOpenAI implementation with no paid requests."""
    row = await invoke(
        MODELS[0],
        {"temperature": 0},
        "quality",
        method,
        payload("quality"),
        transport=mock_transport(valid_quality(), method),
    )
    assert row["success"]
    assert row["http_requests"] == 1
    assert not row["fallback"]
    assert row["usage"]["total_tokens"] == 20
    assert "synthetic-test-secret" not in json.dumps(row)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "range"])
async def test_schema_failures_are_visible(failure):
    """Missing fields and out-of-range scores cannot be scored as success."""
    data = valid_quality()
    if failure == "missing":
        data.pop("relevance")
    else:
        data["completeness"] = 9
    row = await invoke(
        MODELS[0],
        {},
        "quality",
        "json_mode",
        payload("quality"),
        transport=mock_transport(data, "json_mode"),
    )
    assert row["api_success"]
    assert not row["validation_success"]
    assert row["missing_fields"] if failure == "missing" else row["range_violation"]


def test_enum_and_coverage_index_invariants():
    """Enums and one-output-per-requirement invariants remain strict."""
    invalid = {
        "requirements": [{"requirement_index": 0, "status": "maybe", "reason": "test"}]
    }
    assert validate_output("coverage", invalid)["enum_violation"]
    assert not validate_output("coverage", {"requirements": []})["validation_success"]


@pytest.mark.asyncio
async def test_missing_usage_and_extra_content_are_not_hidden():
    """A schema-valid tool response can contain extra text and absent usage."""
    row = await invoke(
        MODELS[0],
        {},
        "quality",
        "function_calling",
        payload("quality"),
        transport=mock_transport(
            valid_quality(), "function_calling", extra_text=True, usage=False
        ),
    )
    assert row["success"] and row["raw_extra_content"]
    assert row["usage"] is None


def test_no_silent_fallback_selection_and_position_mapping():
    """Fallback successes are ineligible and swaps map to the same logical answer."""
    matrix = [{"requested_method": "json_schema", "success": True} for _ in range(2)]
    assert select_method(matrix) is None
    assert select_method([{**row, "success": False} for row in matrix]) is None
    successful = [
        {
            "kind": kind,
            "requested_method": "json_schema",
            "success": True,
            "fallback": False,
        }
        for kind in ("coverage", "quality", "pairwise")
    ]
    assert select_method(successful) == "json_schema"
    successful[0]["fallback"] = True
    assert select_method(successful) is None
    assert logical_winner({"parsed": {"winner": "B"}, "swapped": True}) == "A"
    assert logical_winner({"parsed": {"winner": "A"}, "swapped": False}) == "A"
    assert rates([{"success": True}, {"success": False}])["success"]["rate"] == 0.5


def test_secret_scrubbing_and_long_inputs():
    """Artifacts redact credentials and test the actual forty-thousand-char envelope."""
    value = scrub(
        {"authorization": "hidden", "error": "contains synthetic-test-secret"},
        ("synthetic-test-secret",),
    )
    assert (
        "authorization" not in value and "synthetic-test-secret" not in value["error"]
    )
    inputs = payload("quality", "A", long=True)
    assert sum(len(item["text"]) for item in inputs["evidence_context"]) == 20000
    assert len(inputs["final_report"]) == 20000


@pytest.mark.asyncio
async def test_provider_error_is_recorded_without_retry():
    """HTTP failures remain observable and cannot trigger a hidden retry."""
    calls = []

    async def handle(request):
        calls.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "bad synthetic-test-secret",
                    "type": "invalid_request_error",
                }
            },
        )

    row = await invoke(
        MODELS[0],
        {},
        "quality",
        "json_schema",
        payload("quality"),
        transport=httpx.MockTransport(handle),
    )
    assert row["http_status"] == 400 and not row["success"]
    assert len(calls) == row["http_requests"] == 1
    assert "synthetic-test-secret" not in json.dumps(row)


@pytest.mark.asyncio
async def test_frozen_completed_acceptance_never_reissues_requests(
    tmp_path, monkeypatch
):
    """Reuse a reviewed acceptance result without overwriting or paid requests."""
    from eval.common import atomic_json, sha256
    from eval.judge_acceptance import Acceptance, run
    from eval.judge_synthetic import acceptance_hashes, snapshot

    profile = {"model": "synthetic-reviewed", "applied_to_environment": False}
    atomic_json(tmp_path / "recommended_config.json", profile)
    digest = sha256((tmp_path / "recommended_config.json").read_bytes())
    atomic_json(
        tmp_path / "manifest.json",
        {
            "acceptance_hashes": acceptance_hashes(),
            "protected_hashes": snapshot(),
            "recommended_config_sha256": digest,
            "observed_calls": 0,
        },
    )

    async def forbidden(*args):
        raise AssertionError("Completed acceptance must not call any model")

    monkeypatch.setattr(Acceptance, "model", forbidden)
    await run(tmp_path)
    assert sha256((tmp_path / "recommended_config.json").read_bytes()) == digest
    assert not (tmp_path / "calls").exists()
