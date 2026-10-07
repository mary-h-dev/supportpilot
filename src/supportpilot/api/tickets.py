import json
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import service
from ..db import get_session
from ..logging import log
from ..schemas import DecisionIn, TicketCreate, TicketDetail, TicketOut, TicketStatus
from .auth import require_api_key

router = APIRouter(prefix="/tickets", tags=["tickets"], dependencies=[Depends(require_api_key)])

COLS = "id, sender_email, subject, body, language, category, urgency, status, created_at"


def _runtime(request: Request):
    return getattr(request.app.state, "runtime", None)


@router.post("", response_model=TicketOut, status_code=201)
async def create_ticket(
    data: TicketCreate,
    request: Request,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Stores the ticket and returns immediately; the agent runs in the background.
    Poll GET /tickets/{id}: new -> processing -> awaiting_approval | needs_human."""
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
    if rt := _runtime(request):
        background.add_task(service.process_ticket, rt, row["id"])
    else:
        log.warning("agent_unavailable_ticket_left_new", ticket_id=str(row["id"]))
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


@router.get("/{ticket_id}", response_model=TicketDetail)
async def get_ticket(ticket_id: UUID, session: AsyncSession = Depends(get_session)):
    try:
        return await service.ticket_detail(session, ticket_id)
    except service.NotFound:
        raise HTTPException(404, "ticket not found") from None


@router.post("/{ticket_id}/decision", response_model=TicketDetail)
async def decide(
    ticket_id: UUID,
    body: DecisionIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Human approve / edit / reject. Approve and edit send the (mock) reply; reject does not.
    If sending fails the decision stays recorded: post the same decision again to retry."""
    rt = _runtime(request)
    if rt is None:
        raise HTTPException(503, "agent runtime unavailable (is LLM_API_KEY set?)")
    try:
        await service.decide(rt, ticket_id, body)
    except service.NotFound:
        raise HTTPException(404, "ticket not found") from None
    except service.Conflict as e:
        raise HTTPException(409, str(e)) from None
    except Exception:
        log.exception("decision_failed", ticket_id=str(ticket_id))
        raise HTTPException(
            502, "decision recorded but sending failed; POST the same decision again to retry"
        ) from None
    return await service.ticket_detail(session, ticket_id)
