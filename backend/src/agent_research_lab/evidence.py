"""Evidence normalization and report-gating helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlparse

URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]}]+")
MIN_RESEARCH_BRIEF_LENGTH = 80


def normalize_url(value: str) -> str | None:
    """Return a clean HTTP(S) URL, or ``None`` for non-web content."""
    candidate = value.strip().rstrip(".,;:!?，。；：！")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


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
