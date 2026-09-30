from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
