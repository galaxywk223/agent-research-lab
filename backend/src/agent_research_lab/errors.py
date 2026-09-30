"""Domain errors surfaced by the research workflow."""


class InvalidResearchBriefError(ValueError):
    """Raised when the generated research brief is too short to be reliable."""


class MissingResearchEvidenceError(RuntimeError):
    """Raised when a report would be generated without verified search evidence."""
