"""kb_chunks: search over normalized text

`content` keeps the original text (shown to humans / cited).
`content_norm` is the normalized text (Persian/Arabic letter unification, digits,
ZWNJ...) used for BOTH the tsvector and the embedding, so query and index match.
"""

from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_tsv")
    op.execute("ALTER TABLE kb_chunks DROP COLUMN tsv")
    op.execute("ALTER TABLE kb_chunks ADD COLUMN content_norm TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE kb_chunks ALTER COLUMN content_norm DROP DEFAULT")
    op.execute(
        "ALTER TABLE kb_chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', content_norm)) STORED"
    )
    op.execute("CREATE INDEX ix_kb_chunks_tsv ON kb_chunks USING gin(tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_tsv")
    op.execute("ALTER TABLE kb_chunks DROP COLUMN tsv")
    op.execute("ALTER TABLE kb_chunks DROP COLUMN content_norm")
    op.execute(
        "ALTER TABLE kb_chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED"
    )
    op.execute("CREATE INDEX ix_kb_chunks_tsv ON kb_chunks USING gin(tsv)")
