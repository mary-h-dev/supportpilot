"""drafts: approval state, enforced by the database itself

A draft can only be 'approved' or 'sent' if a decision (who + when) was recorded.
send_reply (MCP) additionally requires status='approved' in one atomic UPDATE.
"""

from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.execute("""
    ALTER TABLE drafts
        ADD COLUMN final_text TEXT,          -- text actually approved (original or human-edited)
        ADD COLUMN decided_at TIMESTAMPTZ,
        ADD COLUMN decided_by TEXT,
        ADD COLUMN sent_at TIMESTAMPTZ""")
    op.execute("""
    ALTER TABLE drafts ADD CONSTRAINT drafts_status_chk
        CHECK (status IN ('pending','approved','rejected','sent','superseded'))""")
    op.execute("""
    ALTER TABLE drafts ADD CONSTRAINT drafts_approved_needs_decision
        CHECK (status NOT IN ('approved','sent')
               OR (decided_at IS NOT NULL AND decided_by IS NOT NULL))""")
    op.execute("""
    ALTER TABLE drafts ADD CONSTRAINT drafts_sent_needs_sent_at
        CHECK (status <> 'sent' OR sent_at IS NOT NULL)""")
    op.execute("CREATE INDEX ix_drafts_ticket ON drafts(ticket_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_drafts_ticket")
    for c in ("drafts_sent_needs_sent_at", "drafts_approved_needs_decision", "drafts_status_chk"):
        op.execute(f"ALTER TABLE drafts DROP CONSTRAINT IF EXISTS {c}")
    op.execute(
        "ALTER TABLE drafts DROP COLUMN final_text, DROP COLUMN decided_at, "
        "DROP COLUMN decided_by, DROP COLUMN sent_at"
    )
