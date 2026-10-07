from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TicketStatus(StrEnum):
    NEW = "new"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    NEEDS_HUMAN = "needs_human"
    SENT = "sent"
    REJECTED = "rejected"


class TicketCreate(BaseModel):
    sender_email: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=20_000)


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sender_email: str
    subject: str
    body: str
    language: str | None
    category: str | None
    urgency: str | None
    status: TicketStatus
    created_at: datetime


class Decision(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class DecisionIn(BaseModel):
    decision: Decision
    reviewer: str = Field(min_length=1, max_length=200)  # self-declared (see limitations)
    edited_text: str | None = Field(default=None, max_length=10_000)
    reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _edit_needs_text(self):
        if self.decision == Decision.EDIT and not (self.edited_text or "").strip():
            raise ValueError("edited_text is required when decision is 'edit'")
        return self


class DraftOut(BaseModel):
    id: UUID
    text: str  # what the agent wrote
    final_text: str | None  # what the human approved (only when edited)
    sources: list[dict]
    confidence: float
    status: str
    decided_by: str | None
    decided_at: datetime | None


class AuditEventOut(BaseModel):
    event: str
    payload: dict
    created_at: datetime


class TicketDetail(TicketOut):
    draft: DraftOut | None = None
    agent: dict | None = None  # {outcome, reason, flags, cost_usd, ...} from the last agent run
    audit: list[AuditEventOut] = []
