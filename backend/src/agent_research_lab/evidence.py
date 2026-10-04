"""Evidence normalization and report-gating helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]}]+")
MIN_RESEARCH_BRIEF_LENGTH = 80


def normalize_url(value: str) -> str | None:
    """Return a clean HTTP(S) URL, or ``None`` for non-web content."""
    candidate = value.strip().rstrip(".,;:!?，。；：！")
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    hostname = parsed.hostname
    if not hostname:
        return None
    host = hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = parsed.port
    if port and not ((parsed.scheme.lower() == "http" and port == 80) or (parsed.scheme.lower() == "https" and port == 443)):
        host = f"{host}:{port}"
    username = parsed.username
    password = parsed.password
    auth = ""
    if username is not None:
        auth = username
        if password is not None:
            auth += f":{password}"
        auth += "@"
    return urlunsplit((parsed.scheme.lower(), f"{auth}{host}", parsed.path, parsed.query, ""))


def canonical_content(value: str | None) -> str | None:
    """Normalize evidence text before hashing and snapshot storage."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized


def content_hash(content: str | None, snippet: str | None = None) -> str | None:
    """Return the SHA-256 hash of the canonical evidence content."""
    canonical = canonical_content(content) or canonical_content(snippet)
    if not canonical:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_source_urls(observations: Iterable[str]) -> list[str]:
    """Extract stable, deduplicated source URLs from search observations."""
    seen: set[str] = set()
    result: list[str] = []
    for observation in observations:
        for match in URL_PATTERN.findall(str(observation)):
            url = normalize_url(match)
            if url and url not in seen:
                seen.add(url)
                result.append(url)
    return result


def is_valid_research_brief(brief: str | None) -> bool:
    """Require a sufficiently detailed research brief before web retrieval."""
    return bool(brief and len(brief.strip()) >= MIN_RESEARCH_BRIEF_LENGTH)


def append_sources(report: str, source_urls: Iterable[str]) -> str:
    """Append verified sources when a model omitted them from its report."""
    urls = list(dict.fromkeys(source_urls))
    if not urls or any(url in report for url in urls):
        return report
    sources = "\n".join(f"- {url}" for url in urls)
    return f"{report.rstrip()}\n\n### 检索来源\n{sources}"
