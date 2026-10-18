"""Read-only business artifacts and honestly labeled Judge evidence packets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent_research_lab.database import database_session
from agent_research_lab.models import (
    Evidence,
    Session,
    SessionRun,
    Task,
    TaskAttempt,
    task_attempt_evidence,
)
from sqlalchemy import func, select

from eval.common import sha256


def compact_business(business: dict[str, Any]) -> dict[str, Any]:
    """Keep the Judge packet and reference provenance without duplicating Task notes."""
    return {
        **business,
        "evidence_references": [
            {
                key: value
                for key, value in reference.items()
                if key not in {"content", "snippet"}
            }
            for reference in business.get("evidence_references", [])
        ],
        "reference_content_storage": "omitted from per-URL references; frozen bounded evidence_packet retained; immutable full content remains in PostgreSQL",
        "collection_source_sha256": sha256(Path(__file__).read_bytes()),
    }


def evidence_packet(
    references: list[dict[str, Any]], per_item: int = 1000, total: int = 20000
) -> dict:
    """Deduplicate notes and prefer captured Tavily originals with provenance."""
    grouped: dict[tuple, dict] = {}
    for reference in sorted(
        references,
        key=lambda item: (
            item["normalized_url"],
            item["task_id"],
            item["task_attempt_id"],
        ),
    ):
        provenance = reference.get("provenance") or {}
        raw = (
            provenance.get("provider") == "tavily"
            and provenance.get("content_kind") in {"raw_content", "snippet"}
            and bool(provenance.get("captured_at"))
        )
        kind = (
            f"tavily_{provenance['content_kind']}"
            if raw
            else "retrieved_research_notes"
        )
        text = reference.get("content") or reference.get("snippet") or ""
        digest = (
            reference.get("note_digest")
            or hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
        key = (reference["task_id"], reference["task_attempt_id"], digest, kind)
        if key not in grouped:
            grouped[key] = {
                "task_id": reference["task_id"],
                "task_attempt_id": reference["task_attempt_id"],
                "content_hash": digest,
                "context_kind": kind,
                "source_urls": [],
                "evidence_ids": [],
                "_text": text,
                "_original_char_count": reference.get("note_char_count", len(text)),
            }
        item = grouped[key]
        item["source_urls"].append(reference["normalized_url"])
        item["evidence_ids"].append(reference["evidence_id"])
    packets = list(grouped.values())
    raw_urls = {
        url
        for item in packets
        if item["context_kind"] != "retrieved_research_notes"
        for url in item["source_urls"]
    }
    for item in packets:
        if item["context_kind"] == "retrieved_research_notes":
            item["source_urls"] = [
                url for url in item["source_urls"] if url not in raw_urls
            ]
    packets = [item for item in packets if item["source_urls"]]
    packets.sort(
        key=lambda item: (
            item["context_kind"] == "retrieved_research_notes",
            sorted(item["source_urls"]),
            item["task_id"],
            item["task_attempt_id"],
        )
    )
    remaining = total
    for item in packets:
        text = item.pop("_text")
        item["source_urls"] = sorted(set(item["source_urls"]))
        item["evidence_ids"] = sorted(set(item["evidence_ids"]))
        item["original_char_count"] = item.pop("_original_char_count")
        item["text"] = text[: min(per_item, remaining)]
        item["truncated"] = len(item["text"]) < item["original_char_count"]
        remaining -= len(item["text"])
    kinds = {item["context_kind"] for item in packets}
    context = (
        next(iter(kinds))
        if len(kinds) == 1
        else "mixed"
        if kinds
        else "retrieved_research_notes"
    )
    return {
        "evidence_context_kind": context,
        "original_source_verification": False,
        "items": packets,
        "char_count": total - remaining,
        "limits": {"per_item": per_item, "total": total},
        "limitation": "URL links associate Task notes with retrieved sources; notes are not per-webpage original fact transcripts.",
    }


async def collect_business(run_id: str) -> dict[str, Any]:
    """Read the Session, Run, Evidence links, and counts without business mutations."""
    async with database_session() as db:
        pair = (
            await db.execute(
                select(SessionRun, Session)
                .join(Session, Session.id == SessionRun.session_id)
                .where(SessionRun.run_id == run_id)
            )
        ).first()
        if pair is None:
            raise LookupError("Business SessionRun not yet available")
        run, session = pair
        note_text = func.coalesce(
            func.nullif(Evidence.content, ""), Evidence.snippet, ""
        )
        rows = (
            (
                await db.execute(
                    select(
                        Evidence.id.label("evidence_id"),
                        Evidence.url,
                        Evidence.normalized_url,
                        Evidence.title,
                        Evidence.content_hash,
                        Evidence.retrieved_at,
                        Evidence.metadata_.label("metadata"),
                        Task.id.label("task_id"),
                        TaskAttempt.id.label("task_attempt_id"),
                        TaskAttempt.attempt_no,
                        func.substr(note_text, 1, 1000).label("text_excerpt"),
                        func.char_length(note_text).label("note_char_count"),
                        func.encode(
                            func.sha256(func.convert_to(note_text, "UTF8")), "hex"
                        ).label("note_digest"),
                    )
                    .join(
                        task_attempt_evidence,
                        Evidence.id == task_attempt_evidence.c.evidence_id,
                    )
                    .join(
                        TaskAttempt,
                        TaskAttempt.id == task_attempt_evidence.c.task_attempt_id,
                    )
                    .join(Task, Task.id == TaskAttempt.task_id)
                    .where(Evidence.session_id == session.id)
                )
            )
            .mappings()
            .all()
        )
        references = []
        for row in rows:
            provenance = (row["metadata"] or {}).get("provenance") or {}
            references.append(
                {
                    "evidence_id": str(row["evidence_id"]),
                    "task_id": str(row["task_id"]),
                    "task_attempt_id": str(row["task_attempt_id"]),
                    "attempt_no": row["attempt_no"],
                    "url": row["url"],
                    "normalized_url": row["normalized_url"],
                    "title": row["title"],
                    "content": row["text_excerpt"],
                    "note_digest": row["note_digest"],
                    "note_char_count": row["note_char_count"],
                    "content_hash": row["content_hash"],
                    "retrieved_at": row["retrieved_at"].isoformat(),
                    "provenance": {
                        key: provenance.get(key)
                        for key in ("provider", "content_kind", "captured_at")
                    },
                }
            )
        counts = {
            "sessions": 1,
            "session_runs": await db.scalar(
                select(func.count())
                .select_from(SessionRun)
                .where(SessionRun.session_id == session.id)
            ),
            "tasks": await db.scalar(
                select(func.count())
                .select_from(Task)
                .where(Task.session_id == session.id)
            ),
            "attempts": await db.scalar(
                select(func.count())
                .select_from(TaskAttempt)
                .join(Task)
                .where(Task.session_id == session.id)
            ),
            "evidence": await db.scalar(
                select(func.count())
                .select_from(Evidence)
                .where(Evidence.session_id == session.id)
            ),
            "task_attempt_evidence": len(references),
        }
        task_rows = (
            await db.execute(
                select(Task.id, Task.topic, Task.status)
                .where(Task.session_id == session.id)
                .order_by(Task.task_order, Task.id)
            )
        ).all()
        attempt_rows = (
            await db.execute(
                select(
                    TaskAttempt.id,
                    TaskAttempt.task_id,
                    TaskAttempt.attempt_no,
                    TaskAttempt.status,
                    TaskAttempt.error.is_not(None).label("error_present"),
                )
                .join(Task)
                .where(Task.session_id == session.id)
                .order_by(TaskAttempt.task_id, TaskAttempt.attempt_no)
            )
        ).all()
        return compact_business(
            {
                "session_id": str(session.id),
                "research_id": session.research_id,
                "run_id": run.run_id,
                "session_status": getattr(session.status, "value", session.status),
                "session_run_status": getattr(run.status, "value", run.status),
                "final_report": session.final_report,
                "evidence_references": references,
                "source_urls": sorted({item["normalized_url"] for item in references}),
                "database_counts": counts,
                "persisted_source_url_count": counts["evidence"],
                "task_attempt_evidence_association_count": counts["task_attempt_evidence"],
                "task_records": [
                    {
                        "task_id": str(row.id),
                        "research_topic": row.topic,
                        "status": row.status,
                    }
                    for row in task_rows
                ],
                "attempt_records": [
                    {
                        "attempt_id": str(row.id),
                        "task_id": str(row.task_id),
                        "attempt_no": row.attempt_no,
                        "status": row.status,
                        "error_present": row.error_present,
                    }
                    for row in attempt_rows
                ],
                "evidence_packet": evidence_packet(references),
            }
        )
