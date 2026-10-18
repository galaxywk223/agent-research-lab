"""Blind offline Coverage, research-note Quality, and Pairwise judges."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Literal

import httpx
from agent_research_lab.metrics import extract_usage
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from eval.common import (
    ROOT,
    Question,
    atomic_json,
    emit,
    frozen_assets,
    paired_order,
    read_json,
    sha256,
    utc_now,
)
from eval.development import (
    DevelopmentQuestion,
    dataset_assets,
    judge_snapshot,
    verify_holdout,
)

DIMENSIONS = (
    "completeness",
    "relevance",
    "evidence_support",
    "synthesis",
    "research_notes_consistency",
)
REASONS = Literal[
    "completeness",
    "relevance",
    "evidence_presentation",
    "synthesis",
    "no_material_difference",
]


class RequirementScore(BaseModel):
    """A coverage judgment for exactly one frozen requirement."""

    model_config = ConfigDict(extra="forbid")
    requirement_index: StrictInt = Field(ge=0)
    status: Literal["covered", "partially_covered", "not_covered"]
    reason: str = Field(min_length=1)


class CoverageOutput(BaseModel):
    """Structured coverage output before programmatic score calculation."""

    model_config = ConfigDict(extra="forbid")
    requirements: list[RequirementScore]


class QualityOutput(BaseModel):
    """Five anchored integer judgments against the supplied context only."""

    model_config = ConfigDict(extra="forbid")
    completeness: StrictInt = Field(ge=1, le=5)
    relevance: StrictInt = Field(ge=1, le=5)
    evidence_support: StrictInt = Field(ge=1, le=5)
    synthesis: StrictInt = Field(ge=1, le=5)
    research_notes_consistency: StrictInt = Field(ge=1, le=5)
    reasons: dict[str, str]

    @model_validator(mode="after")
    def check_reasons(self) -> QualityOutput:
        """Require an explanation for every fixed dimension."""
        if set(self.reasons) != set(DIMENSIONS) or any(
            not text.strip() for text in self.reasons.values()
        ):
            raise ValueError("Reasons must match all five quality dimensions")
        return self


class PairwiseOutput(BaseModel):
    """A blinded preference with fixed reason categories."""

    model_config = ConfigDict(extra="forbid")
    winner: Literal["A", "B", "tie"]
    reason_categories: list[REASONS] = Field(min_length=1)
    reason: str = Field(min_length=1)


class Judge:
    """Explicitly configured real model or deterministic simulated evaluator."""

    def __init__(self, mode: str) -> None:
        """Require an independent model name for real evaluation."""
        if mode not in {"fake", "real"}:
            raise ValueError("Invalid Judge mode")
        self.mode = mode
        self.model_name = (
            "fake-evaluation-v1"
            if mode == "fake"
            else os.getenv("EVAL_JUDGE_MODEL", "").strip()
        )
        if not self.model_name:
            raise ValueError(
                "EVAL_JUDGE_MODEL must be explicitly configured for real evaluation"
            )
        self.profile = judge_snapshot(require_environment=mode == "real")
        self.last_call: dict = {}
        self.transport = None

    async def invoke(
        self, kind: str, payload: dict, schema: type[BaseModel]
    ) -> tuple[BaseModel, dict | None]:
        """Invoke one Judge call; fake responses test plumbing, not answer quality."""
        if self.mode == "fake":
            if kind == "coverage":
                value = {
                    "requirements": [
                        {
                            "requirement_index": index,
                            "status": "partially_covered",
                            "reason": "Simulated fixture; not a quality claim",
                        }
                        for index in range(len(payload["requirements"]))
                    ]
                }
            elif kind == "quality":
                value = {
                    **dict.fromkeys(DIMENSIONS, 3),
                    "reasons": dict.fromkeys(
                        DIMENSIONS, "Simulated fixture; not a quality claim"
                    ),
                }
            else:
                value = {
                    "winner": "tie",
                    "reason_categories": ["no_material_difference"],
                    "reason": "Simulated fixture",
                }
            return schema.model_validate(value), {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
            }
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        from eval.judge_acceptance import WireObserver, response_facts, scrub

        key = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        base = os.getenv("EVAL_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not key or not base:
            raise ValueError("Local Judge credential and base URL required")
        observer = WireObserver((key,))
        self.last_call = {}
        async with httpx.AsyncClient(
            transport=self.transport,
            timeout=180,
            event_hooks={
                "request": [observer.request],
                "response": [observer.response],
            },
        ) as client:
            model = ChatOpenAI(
                model=self.model_name.removeprefix("openai:"),
                api_key=key,
                base_url=base,
                temperature=0,
                reasoning_effort="low",
                max_tokens=8192,
                max_retries=0,
                use_responses_api=False,
                http_async_client=client,
                timeout=180,
            )
            prompt = (ROOT / f"prompts/{kind}_judge_v1.txt").read_text(encoding="utf-8")
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    response = await model.with_structured_output(
                        schema, method="json_schema", include_raw=True
                    ).ainvoke(
                        [
                            SystemMessage(content=prompt),
                            HumanMessage(
                                content=json.dumps(payload, ensure_ascii=False)
                            ),
                        ]
                    )
            finally:
                self.last_call = {
                    "wire_requests": observer.requests,
                    "http_status": observer.responses[-1]["http_status"]
                    if observer.responses
                    else None,
                    "fallback": any(
                        item["method"] != "json_schema" for item in observer.requests
                    ),
                    "response_facts": response_facts(observer),
                    "warnings": scrub([str(item.message) for item in caught], (key,)),
                }
            if self.last_call["fallback"]:
                raise ValueError("Unexpected structured-method fallback rejected")
            if self.last_call["response_facts"]["finish_reason"] == "length":
                raise ValueError("Truncated Judge output rejected")
        if response.get("parsing_error") or response.get("parsed") is None:
            raise ValueError("Judge structured-output parsing failed")
        return response["parsed"], extract_usage(response.get("raw"))


def blind_payload(question: Question, result: dict, rubric: dict) -> dict:
    """Build Judge input with no strategy, cost, experiment or model metadata."""
    packet = (
        result.get("evidence_packet")
        or (result.get("business") or {}).get("evidence_packet")
        or {}
    )
    return {
        "question": question.question,
        "requirements": question.requirements,
        "final_report": result.get("final_report", ""),
        "rubric": rubric,
        "evidence_context_kind": packet.get(
            "evidence_context_kind", "retrieved_research_notes"
        ),
        "evidence_context": [
            {
                key: item.get(key)
                for key in ("text", "context_kind", "source_urls", "truncated")
            }
            for item in packet.get("items", [])
        ],
    }


async def judge_call(
    judge: Judge,
    kind: str,
    payload: dict,
    schema: type[BaseModel],
    journal: Path | None = None,
) -> dict:
    """Keep failed Judge calls explicit and separate from Research costs."""
    payload = {**payload, "output_json_schema": schema.model_json_schema()}
    started = time.perf_counter()
    result = {
        "status": "failed",
        "output": None,
        "error": None,
        "usage": None,
        "model": judge.model_name,
        "mode": judge.mode,
        "temperature": 0,
        "temperature_effective": "unknown",
        "structured_method": "json_schema",
        "reasoning_effort": "low",
        "max_tokens": 8192,
        "max_retries": 0,
        "usage_source": "simulated"
        if judge.mode == "fake" or judge.transport is not None
        else "provider",
        "retry_usage_included": "unknown",
        "prompt_version": "v1",
        "started_at": utc_now(),
        "call_id": str(uuid.uuid4()),
        "prompt_sha256": sha256((ROOT / f"prompts/{kind}_judge_v1.txt").read_bytes()),
        "input_sha256": sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ),
    }
    identity = {
        "input_sha256": result["input_sha256"],
        "judge_config": sha256((ROOT / "configs/judge_phase2_v1.json").read_bytes()),
        "model": judge.model_name,
        "mode": judge.mode,
        "prompt_sha256": result["prompt_sha256"],
    }
    if journal and journal.exists():
        prior = read_json(journal)
        if prior["identity"] != identity:
            raise ValueError("Judge input/config changed for an existing call")
        if prior["status"] == "finished":
            return prior["result"]
        result.update(
            call_id=prior.get("call_id"),
            started_at=prior.get("started_at"),
            latency_ms=None,
            finished_at=None,
        )
        result["error"] = {
            "type": "UnknownJudgeOutcome",
            "message": "Interrupted paid call retained; no automatic resend",
        }
        return result
    if journal:
        atomic_json(
            journal,
            {
                "status": "starting",
                "identity": identity,
                "started_at": result["started_at"],
                "call_id": result["call_id"],
            },
        )
    try:
        output, usage = await judge.invoke(kind, payload, schema)
        output = schema.model_validate(output)
        result.update(status="completed", output=output.model_dump(), usage=usage)
    except Exception as exc:
        result["usage"] = (judge.last_call.get("response_facts") or {}).get("usage")
        result["error"] = {
            "type": type(exc).__name__,
            "message": "Judge call or structured validation failed",
        }
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["finished_at"] = utc_now()
    result["wire_observation"] = judge.last_call
    if journal:
        atomic_json(
            journal, {"status": "finished", "identity": identity, "result": result}
        )
    return result


async def evaluate_result(
    question: Question | DevelopmentQuestion,
    result: dict,
    judge: Judge,
    journal_dir: Path | None = None,
) -> dict:
    """Evaluate a terminal report with fixed coverage and quality scoring rules."""
    if isinstance(question, DevelopmentQuestion):
        verify_holdout()
        if result.get("dataset_split") != "development":
            raise ValueError("Development result split missing or mismatched")
    elif judge.mode == "real":
        raise ValueError("Real holdout Judge execution is locked")
    else:
        frozen_assets()
    outcome: dict[str, Any] = {
        "coverage_score": None,
        "quality_score": None,
        "judge_mode": judge.mode,
        "judge_model": judge.model_name,
        "started_at": utc_now(),
        "status": "skipped",
        "rubric_sha256": sha256((ROOT / "rubric_v1.json").read_bytes()),
    }
    if (
        result.get("status") != "completed"
        or not result.get("final_report")
        or result.get("config_valid") is not True
    ):
        outcome["reason"] = "Research incomplete or configuration invalid"
        return outcome
    rubric = read_json(ROOT / "rubric_v1.json")
    payload = blind_payload(question, result, rubric)
    coverage = await judge_call(
        judge,
        "coverage",
        {key: payload[key] for key in ("question", "requirements", "final_report")},
        CoverageOutput,
        journal_dir / "coverage.json" if journal_dir else None,
    )
    if coverage["status"] == "completed":
        scores = coverage["output"]["requirements"]
        if sorted(item["requirement_index"] for item in scores) != list(
            range(len(question.requirements))
        ):
            coverage.update(
                status="failed",
                error={
                    "type": "CoverageIndicesError",
                    "message": "Each frozen requirement must be scored exactly once",
                },
            )
        else:
            outcome["coverage_score"] = sum(
                rubric["coverage"][item["status"]] for item in scores
            ) / len(scores)
    quality = await judge_call(
        judge,
        "quality",
        payload,
        QualityOutput,
        journal_dir / "quality.json" if journal_dir else None,
    )
    if quality["status"] == "completed":
        outcome["quality_score"] = sum(quality["output"][key] for key in DIMENSIONS) / 5
    outcome.update(
        coverage=coverage,
        quality=quality,
        finished_at=utc_now(),
        status="completed"
        if coverage["status"] == quality["status"] == "completed"
        else "failed",
        evidence_context_kind=payload["evidence_context_kind"],
        original_source_verification=False,
    )
    return outcome


async def evaluate_pair(
    question: Question, first: dict, second: dict, judge: Judge
) -> dict:
    """Present a seeded blind pair without running either research strategy."""
    if judge.mode == "real":
        raise ValueError("Real holdout pairwise execution is locked")
    _, manifest, _ = frozen_assets()
    answers = {first["strategy"]: first, second["strategy"]: second}
    if set(answers) != {"baseline_fixed", "adaptive"}:
        raise ValueError("Pair requires one baseline_fixed and one adaptive result")
    for result in answers.values():
        if (
            result["question_id"] != question.question_id
            or result["questions_sha256"] != manifest["questions_sha256"]
            or result["status"] != "completed"
        ):
            raise ValueError(
                "Pair identity or completion does not match frozen protocol"
            )
    order = paired_order(question.question_id, "presentation")
    output = await judge_call(
        judge,
        "pairwise",
        {
            "question": question.question,
            "requirements": question.requirements,
            "answer_A": answers[order[0]]["final_report"],
            "answer_B": answers[order[1]]["final_report"],
        },
        PairwiseOutput,
    )
    winner = (output.get("output") or {}).get("winner")
    output.update(
        presentation_order=order,
        execution_order_planned=paired_order(question.question_id),
        winner_strategy=order[0]
        if winner == "A"
        else order[1]
        if winner == "B"
        else winner,
        question_id=question.question_id,
    )
    return output


async def evaluate_experiment(directory: Path, mode: str) -> dict:
    """Evaluate existing completed reports and preserve prior evaluator attempts."""
    split = (read_json(directory / "experiment_manifest.json")).get(
        "dataset_split", "evaluation"
    )
    questions, frozen, _ = dataset_assets(split, "real" if mode == "real" else "fake")
    judge = Judge(mode)
    path = directory / "experiment_manifest.json"
    manifest = read_json(path)
    profile = judge_snapshot(require_environment=mode == "real")
    if manifest.get("provider_mode") != mode:
        raise ValueError("Judge and Research provider modes cannot be mixed")
    expected_protocol = {
        "dataset": frozen["questions_sha256"],
        "baseline": sha256((ROOT / "configs/baseline_v1.json").read_bytes()),
        "judge": sha256((ROOT / "configs/judge_phase2_v1.json").read_bytes()),
        "prompt": profile["prompt_sha256"],
        "rubric": profile["rubric_sha256"],
    }
    if manifest.get("protocol_hashes") != expected_protocol:
        raise ValueError("Experiment Judge/dataset/protocol identity changed")
    judge_config = {
        "model": judge.model_name,
        "mode": mode,
        "structured_method": "json_schema",
        "reasoning_effort": "low",
        "temperature_requested": 0,
        "temperature_effective": "unknown",
        "max_tokens": 8192,
        "max_retries": 0,
        "prompt_sha256": profile["prompt_sha256"],
        "rubric_sha256": profile["rubric_sha256"],
    }
    if (
        manifest.get("judge_execution_config")
        and manifest["judge_execution_config"] != judge_config
    ):
        raise ValueError("Judge execution configuration changed within an experiment")
    manifest["judge_execution_config"] = judge_config
    manifest["judge_evaluator_source_sha256"] = sha256(Path(__file__).read_bytes())
    atomic_json(path, manifest)
    statuses = {}
    for question in questions:
        result_path = directory / f"{question.question_id}.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        if (
            result.get("question_id") != question.question_id
            or result.get("dataset_split") != split
            or result.get("protocol_hashes") != expected_protocol
            or result.get("questions_sha256") != frozen["questions_sha256"]
            or result.get("provider_mode") != mode
        ):
            raise ValueError("Result identity differs from its frozen experiment")
        if result.get("evaluation", {}).get("status") in {"completed", "failed"}:
            statuses[question.question_id] = result["evaluation"]["status"]
            continue
        if result.get("evaluation"):
            result.setdefault("evaluation_history", []).append(result["evaluation"])
        result["evaluation"] = await evaluate_result(
            question,
            result,
            judge,
            directory
            / "judge_calls"
            / question.question_id
            / f"attempt-{result['attempt_no']:04}",
        )
        atomic_json(result_path, result)
        atomic_json(
            directory / "evaluations" / f"{question.question_id}.json",
            result["evaluation"],
        )
        statuses[question.question_id] = result["evaluation"]["status"]
    verify_holdout()
    return statuses


def main() -> None:
    """Run explicit offline evaluation against saved experiment results."""
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    args = parser.parse_args()
    emit(asyncio.run(evaluate_experiment(args.experiment_dir, args.mode)))


if __name__ == "__main__":
    main()
