"""Deterministic guards on a draft. Cheap, explainable, and not fooled by the LLM."""

from .schemas import DraftOutput


def evaluate_draft(draft: DraftOutput, passages: list[dict], min_confidence: float) -> str:
    """Return "" if the draft passes, else a machine-readable reason to route to a human."""
    if not draft.answerable:
        return "kb_has_no_answer"
    if not draft.reply_text.strip():
        return "empty_draft"
    if not draft.cited_source_ids:
        return "no_citations"
    allowed = {p["source_id"] for p in passages}
    if not set(draft.cited_source_ids) <= allowed:
        return "invalid_citation"  # model cited something retrieval never returned
    if draft.confidence < min_confidence:
        return "low_confidence"
    return ""
