"""Development identity, frozen Judge config, and holdout protection."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from eval.common import ROOT, SEED, Question, frozen_assets, read_json, sha256

HOLDOUT_HASH = "9511f18eb766976a115d35768f56ee6910172458e69507082a42da867883bf76"
HOLDOUT_MANIFEST_HASH = (
    "52e1c1317cda678f030f8b1ababc3b0f79d95819b131a0c59fa385cae273fa97"
)


class DevelopmentQuestion(BaseModel):
    """One independent historical research case for strategy development."""

    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(pattern=r"^D0[1-5]$")
    question: str = Field(min_length=10)
    category: str
    requirements: list[str] = Field(min_length=3, max_length=6)
    notes: str
    time_sensitive: bool


def verify_holdout(root: Path = ROOT) -> dict:
    """Verify frozen byte hashes without parsing holdout questions into cases."""
    if sha256((root / "manifest_v1.json").read_bytes()) != HOLDOUT_MANIFEST_HASH:
        raise ValueError("Holdout manifest changed")
    manifest = read_json(root / "manifest_v1.json")
    for name, expected in {
        "questions_v1.jsonl": HOLDOUT_HASH,
        **manifest["artifacts_sha256"],
    }.items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError(f"Holdout artifact changed: {name}")
    return manifest


def judge_snapshot(root: Path = ROOT, require_environment: bool = False) -> dict:
    """Validate the officially approved model and exact synthetic invocation contract."""
    profile = read_json(root / "configs/judge_phase2_v1.json")
    expected = {
        "model": "openai:glm-5.3",
        "structured_method": "json_schema",
        "reasoning_effort": "low",
        "temperature_requested": 0,
        "temperature_effective": "unknown",
        "max_tokens": 8192,
        "max_retries": 0,
        "use_responses_api": False,
        "include_raw": True,
        "enable_thinking": "omitted",
    }
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen Judge parameters changed")
    for kind, expected_hash in profile["prompt_sha256"].items():
        if (
            sha256((root / f"prompts/{kind}_judge_v1.txt").read_bytes())
            != expected_hash
        ):
            raise ValueError("Judge prompt changed")
    if sha256((root / "rubric_v1.json").read_bytes()) != profile["rubric_sha256"]:
        raise ValueError("Judge rubric changed")
    from eval.evaluate import CoverageOutput, PairwiseOutput, QualityOutput

    for kind, schema in {
        "coverage": CoverageOutput,
        "quality": QualityOutput,
        "pairwise": PairwiseOutput,
    }.items():
        digest = sha256(
            json.dumps(
                schema.model_json_schema(), sort_keys=True, ensure_ascii=False
            ).encode()
        )
        if digest != profile["schema_sha256"][kind]:
            raise ValueError(f"Judge schema changed: {kind}")
    if require_environment:
        for key, value in profile["environment_values"].items():
            if os.getenv(key) != value:
                raise ValueError(f"{key} must match the approved Judge configuration")
        for name, expected_version in profile["dependency_versions"].items():
            if version(name) != expected_version:
                raise ValueError(f"Judge dependency version changed: {name}")
    return profile


def dataset_assets(
    split: str, provider_mode: str, root: Path = ROOT
) -> tuple[list[Question | DevelopmentQuestion], dict, dict]:
    """Load only the requested split and reject paid holdout execution."""
    if split not in {"development", "evaluation"}:
        raise ValueError("Explicit dataset_split is required")
    if split == "evaluation":
        if provider_mode != "fake":
            raise ValueError(
                "Holdout real execution is locked until the final paired phase"
            )
        return frozen_assets(root)
    verify_holdout(root)
    manifest = read_json(root / "dev_manifest_v1.json")
    if (
        manifest["dataset_split"] != split
        or manifest["dev_version"] != "dev-v1"
        or manifest["seed"] != SEED
    ):
        raise ValueError("Development identity changed")
    checks = {
        "dev_questions_v1.jsonl": manifest["questions_sha256"],
        "configs/baseline_v1.json": manifest["baseline_config_sha256"],
        "configs/judge_phase2_v1.json": manifest["judge_config_sha256"],
    }
    for name, digest in checks.items():
        if sha256((root / name).read_bytes()) != digest:
            raise ValueError(f"Frozen Development artifact changed: {name}")
    questions = [
        DevelopmentQuestion.model_validate_json(line)
        for line in (root / "dev_questions_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(questions) != manifest["question_count"] or [
        q.question_id for q in questions
    ] != [f"D{i:02}" for i in range(1, 6)]:
        raise ValueError("Development question identities changed")
    if any(
        q.time_sensitive or len(set(q.requirements)) != len(q.requirements)
        for q in questions
    ):
        raise ValueError("Development requirements or time scope invalid")
    judge_snapshot(root)
    return questions, manifest, read_json(root / "configs/baseline_v1.json")
