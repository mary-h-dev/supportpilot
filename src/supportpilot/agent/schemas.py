import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["billing", "refund", "cancellation", "account_access", "technical", "other"]
Urgency = Literal["low", "normal", "high"]
Language = Literal["en", "fa", "mixed"]


class Classification(BaseModel):
    category: Category
    urgency: Urgency
    language: Language


class DraftOutput(BaseModel):
    answerable: bool
    reply_text: str = ""
    cited_source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class FaithfulnessCheck(BaseModel):
    supported: bool  # every factual claim in the draft is backed by the passages
    answers_question: bool  # the draft actually addresses the customer's question
    unsupported_claims: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    ticket: dict  # {id, subject, body, sender_email}
    classification: dict
    passages: list[dict]
    draft: dict | None  # only set when outcome == draft_ready
    discarded_draft: dict | None  # rejected draft, kept for audit/eval; NEVER shown as a draft
    check: dict | None
    flags: list[str]  # e.g. "possible_prompt_injection"
    outcome: Literal["draft_ready", "needs_human"]
    reason: str  # why we routed to needs_human ("" when draft_ready)
    metrics: Annotated[list[dict], operator.add]  # per-step latency/tokens/cost
