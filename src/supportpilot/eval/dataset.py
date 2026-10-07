import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..agent.schemas import Category, Urgency

DEFAULT_PATH = Path("eval/tickets.json")


class EvalTicket(BaseModel):
    id: str
    split: Literal["dev", "test"]
    lang: Literal["en", "fa", "mixed"]
    subject: str
    body: str
    answerable: bool  # False = the KB genuinely has no answer -> correct behaviour is to abstain
    category: Category  # gold labels for the classifier
    urgency: Urgency
    gold_docs: list[str] = Field(default_factory=list)  # KB article slugs (language-agnostic)
    gold_facts: list[str] = Field(default_factory=list)  # facts a correct reply must state
    tags: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)  # strings that must NOT appear in a reply

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable and not (self.gold_docs and self.gold_facts):
            raise ValueError(f"{self.id}: answerable tickets need gold_docs and gold_facts")
        if not self.answerable and (self.gold_docs or self.gold_facts):
            raise ValueError(f"{self.id}: unanswerable tickets must not have gold answers")
        return self


def load_tickets(path: Path = DEFAULT_PATH, split: str | None = None) -> list[EvalTicket]:
    tickets = [EvalTicket(**t) for t in json.loads(Path(path).read_text(encoding="utf-8"))]
    ids = [t.id for t in tickets]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ticket ids")
    return [t for t in tickets if split in (None, t.split)]


def check_against_kb(tickets: list[EvalTicket], kb_root: Path = Path("kb")) -> list[str]:
    """Problems found (empty list = OK): every gold doc must exist in BOTH languages."""
    problems = []
    for t in tickets:
        for slug in t.gold_docs:
            for lang in ("en", "fa"):
                if not (kb_root / lang / f"{slug}.md").exists():
                    problems.append(f"{t.id}: gold doc '{slug}' missing in kb/{lang}/")
    return problems
