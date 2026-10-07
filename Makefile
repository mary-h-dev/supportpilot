.PHONY: up down test lint migrate ingest eval eval-test eval-report
# Windows without `make`? Run the command on the right of each target by hand.

up:      ; docker compose up --build -d
down:    ; docker compose down
test:    ; uv run pytest -q
lint:    ; uv run ruff check . && uv run ruff format --check .
migrate: ; docker compose exec api alembic upgrade head
ingest:  ; docker compose exec api python -m supportpilot.kb.ingest

# Evaluation: needs the stack up (db + Ollama) and LLM_API_KEY in .env.
eval:        ; docker compose exec api python -m supportpilot.eval run --split dev --ingest
# HELD-OUT test set: run once, at the very end. Never tune anything because of it.
eval-test:   ; docker compose exec api python -m supportpilot.eval run --split test --final
# After filling human_decision / human_faithful in the review CSV:
#   make eval-report RESULTS=eval/reports/dev-<ts>.json REVIEWS=eval/reports/review-dev-<ts>.csv
eval-report: ; docker compose exec api python -m supportpilot.eval report $(RESULTS) --reviews $(REVIEWS)
