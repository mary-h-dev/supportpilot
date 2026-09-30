.PHONY: up down test lint migrate
up:      ; docker compose up --build -d
down:    ; docker compose down
test:    ; uv run pytest -q
lint:    ; uv run ruff check . && uv run ruff format --check .
migrate: ; docker compose exec api alembic upgrade head
