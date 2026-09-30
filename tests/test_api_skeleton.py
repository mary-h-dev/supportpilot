import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from supportpilot.db import get_session
from supportpilot.main import app
from supportpilot.schemas import TicketCreate


class FakeSession:
    async def execute(self, *a, **k):
        return None


async def fake_session():
    yield FakeSession()


def test_health_ok():
    app.dependency_overrides[get_session] = fake_session
    try:
        r = TestClient(app).get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}
    finally:
        app.dependency_overrides.clear()


def test_ticket_rejects_empty_body():
    with pytest.raises(ValidationError):
        TicketCreate(sender_email="a@b.com", subject="hi", body="")


def test_ticket_accepts_persian():
    t = TicketCreate(sender_email="a@b.com", subject="مشکل پرداخت", body="پرداخت من انجام نشد")
    assert t.subject == "مشکل پرداخت"
