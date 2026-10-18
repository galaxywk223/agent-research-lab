"""Frozen assets, experiment identity, and atomic evaluation I/O."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
SEED = 20261004
CATEGORIES = Counter(
    {
        "多源事实核验": 1,
        "多实体比较": 2,
        "时间线 / 事件演化": 1,
        "技术方案比较": 1,
        "多子问题综合": 2,
        "冲突证据分析": 1,
        "长尾信息检索": 1,
        "近期信息综合": 1,
    }
)


class Question(BaseModel):
    """One predeclared research question and atomic coverage requirements."""

    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(pattern=r"^Q\d{2}$")
    question: str = Field(min_length=10)
    category: str
    eval_version: str
    requirements: list[str] = Field(min_length=3, max_length=6)
    evaluation_notes: str
    time_sensitive: bool


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256(data: bytes) -> str:
    """Hash exact bytes rather than reserialized JSON."""
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON artifact only after flushing its complete contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".eval-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def frozen_assets(root: Path = ROOT) -> tuple[list[Question], dict, dict]:
    """Verify every frozen artifact before loading the evaluation protocol."""
    manifest = read_json(root / "manifest_v1.json")
    if manifest["seed"] != SEED or manifest["eval_version"] != "v1":
        raise ValueError("Unsupported evaluation identity")
    paths = {
        "questions_v1.jsonl": manifest["questions_sha256"],
        **manifest["artifacts_sha256"],
    }
    for relative, expected in paths.items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or sha256(path.read_bytes()) != expected
        ):
            raise ValueError(f"Frozen artifact hash mismatch: {relative}")
    questions = [
        Question.model_validate_json(line)
        for line in (root / "questions_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(questions) != 10 or manifest["question_count"] != 10:
        raise ValueError("Evaluation v1 requires exactly ten questions")
    if [q.question_id for q in questions] != [f"Q{i:02}" for i in range(1, 11)]:
        raise ValueError("Question identity/order changed")
    if Counter(q.category for q in questions) != CATEGORIES:
        raise ValueError("Question category distribution changed")
    if any(
        q.eval_version != "v1" or len(set(q.requirements)) != len(q.requirements)
        for q in questions
    ):
        raise ValueError("Invalid frozen requirements")
    config = read_json(root / "configs/baseline_v1.json")
    if (
        config["questions_sha256"] != manifest["questions_sha256"]
        or config["seed"] != SEED
    ):
        raise ValueError("Baseline config and question manifest disagree")
    return questions, manifest, config


def paired_order(
    question_id: str, namespace: str = "execution", seed: int = SEED
) -> list[str]:
    """Derive execution or presentation order without consulting results."""
    if namespace not in {"execution", "presentation"}:
        raise ValueError("Unknown order namespace")
    digest = hashlib.sha256(f"{namespace}:v1:{seed}:{question_id}".encode()).digest()
    order = ["baseline_fixed", "adaptive"]
    return order if digest[0] % 2 == 0 else order[::-1]


def temporal_pair_violation(start_a: str, start_b: str) -> bool:
    """Flag pairs whose execution starts are more than sixty minutes apart."""
    return (
        abs(
            (
                datetime.fromisoformat(start_a) - datetime.fromisoformat(start_b)
            ).total_seconds()
        )
        > 3600
    )


def source_identity() -> dict[str, Any]:
    """Record Git and current source hashes without credentials or run outputs."""
    project = ROOT.parent

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project, text=True).strip()

    paths = [
        *project.glob("backend/src/**/*.py"),
        *ROOT.glob("**/*.py"),
        *ROOT.glob("configs/*.json"),
        *ROOT.glob("prompts/*.txt"),
        ROOT / "rubric_v1.json",
        ROOT / "protocol_v1.json",
        ROOT / "questions_v1.jsonl",
        project / "pyproject.toml",
        project / "uv.lock",
    ]
    hashes = {
        p.relative_to(project).as_posix(): sha256(p.read_bytes())
        for p in sorted(set(paths))
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_files_sha256": hashes,
        "source_snapshot_sha256": sha256(json.dumps(hashes, sort_keys=True).encode()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
        },
    }


def emit(value: Any) -> None:
    """Write machine-readable CLI output without including credentials."""
    import sys

    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
