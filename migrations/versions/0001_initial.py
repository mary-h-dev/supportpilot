"""initial schema

Hand-written SQL on purpose: pgvector + generated tsvector are clearer in SQL.
"""

from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("""
    CREATE TABLE tickets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        sender_email TEXT NOT NULL,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        language TEXT,
        category TEXT,
        urgency TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""")
    op.execute("CREATE INDEX ix_tickets_status ON tickets(status)")
    op.execute("""
    CREATE TABLE kb_documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title TEXT NOT NULL,
        source_uri TEXT NOT NULL UNIQUE,
        language TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""")
    op.execute("""
    CREATE TABLE kb_chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
        position INT NOT NULL,
        content TEXT NOT NULL,
        embedding vector(1024),
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
    )""")
    op.execute("CREATE INDEX ix_kb_chunks_tsv ON kb_chunks USING gin(tsv)")
    op.execute(
        "CREATE INDEX ix_kb_chunks_emb ON kb_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("""
    CREATE TABLE drafts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ticket_id UUID NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
        text TEXT NOT NULL,
        sources JSONB NOT NULL DEFAULT '[]',
        confidence DOUBLE PRECISION NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""")
    op.execute("""
    CREATE TABLE audit_log (
        id BIGSERIAL PRIMARY KEY,
        ticket_id UUID REFERENCES tickets(id) ON DELETE SET NULL,
        event TEXT NOT NULL,
        payload JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""")
    op.execute("CREATE INDEX ix_audit_ticket ON audit_log(ticket_id)")


def downgrade() -> None:
    for t in ("audit_log", "drafts", "kb_chunks", "kb_documents", "tickets"):
        op.execute(f"DROP TABLE IF EXISTS {t}")
