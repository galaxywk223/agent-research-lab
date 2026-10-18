"""Pure tests for persistence identity and evidence normalization."""

from agent_research_lab.evidence import content_hash, normalize_url
from agent_research_lab.repository import canonical_task_payload, task_idempotency_key


def test_url_normalization_removes_default_port_and_fragment():
    """Equivalent HTTP URLs normalize to one snapshot key."""
    assert normalize_url("HTTPS://Example.com:443/paper#section") == "https://example.com/paper"


def test_content_hash_is_canonical_sha256():
    """Whitespace and line endings do not change the content hash."""
    assert content_hash("A\r\nB") == content_hash(" A\nB ")
    assert len(content_hash("A") or "") == 64


def test_task_identity_uses_canonical_all_fields():
    """Task identity is stable for canonical arguments and changes with identity fields."""
    assert canonical_task_payload({"research_topic": " a  b "}) == canonical_task_payload({"research_topic": "a b"})
    assert task_idempotency_key({"research_topic": "a", "scope": "one"}) != task_idempotency_key({"research_topic": "a", "scope": "two"})
