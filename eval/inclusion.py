"""Immutable Development Run inclusion and exclusion identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.common import ROOT, atomic_json, read_json, sha256, source_identity, utc_now

INCLUSION_VERSION = "dev-v2-attributed-inclusion-v1"


def manifest_path(directory: Path) -> Path:
    """Return the immutable inclusion manifest path."""
    return directory / "inclusion_manifest.json"


def build_inclusion_manifest(directory: Path) -> dict[str, Any]:
    """Build the fixed valid/excluded set from the recorded audit identities."""
    audit_dir = ROOT / "runs" / "phase23b-characterization"
    status = read_json(audit_dir / "database_status_audit.json")
    by_run = {row["run_id"]: row for row in status}
    valid_ids = {
        "D03": "01a1070a-a3c9-75d2-9d9f-823d5958c699",
        "D04": "01a10716-5b1c-7730-88e0-5b4fbfce50b0",
    }
    excluded_ids = [
        "01a106fc-357d-7e90-a82f-ebc7ece1405d",
        "01a1072f-865f-7eb2-836b-71a479082b9b",
    ]
    entries = []
    for question_id, run_id in valid_ids.items():
        row = by_run[run_id]
        entries.append(
            {
                "question_id": question_id,
                "classification": "valid",
                **row,
                "reason": "unique completed v2 baseline Research",
            }
        )
    reasons = {
        excluded_ids[
            0
        ]: "initialization failure; no provider invocation; not a valid sample",
        excluded_ids[
            1
        ]: "accidental duplicate D04 execution; excluded from characterization",
    }
    for run_id in excluded_ids:
        row = by_run[run_id]
        entries.append(
            {
                "question_id": "D03" if run_id == excluded_ids[0] else "D04",
                "classification": "excluded",
                **row,
                "reason": reasons[run_id],
            }
        )
    identity = source_identity()
    protocol = {
        "dataset": sha256((ROOT / "dev_questions_v1.jsonl").read_bytes()),
        "baseline": sha256((ROOT / "configs/baseline_v1.json").read_bytes()),
        "judge": sha256((ROOT / "configs/judge_phase2_v1.json").read_bytes()),
        "protocol": sha256((ROOT / "protocol_v1.json").read_bytes()),
    }
    return {
        "schema_version": 1,
        "inclusion_version": INCLUSION_VERSION,
        "created_at": utc_now(),
        "dataset_split": "development",
        "purpose": "strategy_development",
        "valid_run_ids": list(valid_ids.values()),
        "excluded_run_ids": excluded_ids,
        "entries": entries,
        "protocol_hashes": protocol,
        "source_snapshot_sha256": identity["source_snapshot_sha256"],
        "immutable": True,
    }


def ensure_inclusion_manifest(directory: Path) -> dict[str, Any]:
    """Create once, then require byte identity on every subsequent read."""
    path = manifest_path(directory)
    if not path.exists():
        value = build_inclusion_manifest(directory)
        atomic_json(path, value)
        return value
    value = read_json(path)
    if (
        value.get("immutable") is not True
        or value.get("inclusion_version") != INCLUSION_VERSION
    ):
        raise ValueError("Invalid inclusion manifest")
    expected = build_inclusion_manifest(directory)
    for key in ("valid_run_ids", "excluded_run_ids", "protocol_hashes"):
        if value.get(key) != expected.get(key):
            raise ValueError(
                "Inclusion manifest identity changed; create a new version"
            )
    return value


def valid_records(directory: Path) -> list[dict]:
    """Load only the explicitly included result files."""
    if not manifest_path(directory).exists() and not (directory / "D03.json").exists():
        return []
    manifest = ensure_inclusion_manifest(directory)
    records = []
    for entry in manifest["entries"]:
        if entry["classification"] != "valid":
            continue
        path = directory / f"{entry['question_id']}.json"
        if not path.exists():
            raise ValueError(f"Included Run artifact missing: {entry['question_id']}")
        row = read_json(path)
        if row.get("run_id") != entry["run_id"] or row.get("status") != "completed":
            raise ValueError("Included Run identity or terminal status changed")
        records.append(row)
    return records
