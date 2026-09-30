import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..logging import log
from ..schemas import TicketCreate, TicketOut, TicketStatus

router = APIRouter(prefix="/tickets", tags=["tickets"])

COLS = "id, sender_email, subject, body, language, category, urgency, status, created_at"


@router.post("", response_model=TicketOut, status_code=201)
async def create_ticket(data: TicketCreate, session: AsyncSession = Depends(get_session)):
    row = (
        (
            await session.execute(
                text(
                    f"INSERT INTO tickets (sender_email, subject, body) "
                    f"VALUES (:e, :s, :b) RETURNING {COLS}"
                ),
                {"e": data.sender_email, "s": data.subject, "b": data.body},
            )
        )
        .mappings()
        .one()
    )
    await session.execute(
        text("INSERT INTO audit_log (ticket_id, event, payload) VALUES (:t, 'ticket_created', :p)"),
        {"t": row["id"], "p": json.dumps({"sender": data.sender_email})},
    )
    await session.commit()
    log.info("ticket_created", ticket_id=str(row["id"]))
    return row


@router.get("", response_model=list[TicketOut])
async def list_tickets(
    status: TicketStatus | None = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    q = f"SELECT {COLS} FROM tickets"
    params: dict = {"limit": min(limit, 200)}
    if status:
        q += " WHERE status = :st"
        params["st"] = status.value
    q += " ORDER BY created_at DESC LIMIT :limit"
    return (await session.execute(text(q), params)).mappings().all()


@router.get("/{ticket_id}", response_model=TicketOut)
async def get_ticket(ticket_id: UUID, session: AsyncSession = Depends(get_session)):
    row = (
        (await session.execute(text(f"SELECT {COLS} FROM tickets WHERE id = :i"), {"i": ticket_id}))
        .mappings()
        .first()
    )
    if not row:
        raise HTTPException(404, "ticket not found")
    return row
