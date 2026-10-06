import hashlib

from supportpilot.kb.normalize import query_terms
from supportpilot.llm import Usage


class FakeLLM:
    """Scripted LLM: one response (dict | callable(user)->dict | Exception) per step name."""

    def __init__(self, **by_step):
        self.by_step, self.calls = by_step, []

    async def complete_json(self, *, step, model, system, user, schema):
        self.calls.append({"step": step, "system": system, "user": user})
        r = self.by_step[step]
        if isinstance(r, Exception):
            raise r
        if callable(r):
            r = r(user)
        usage = Usage(
            model=model, prompt_tokens=10, completion_tokens=5, latency_ms=1.0, cost_usd=0.0001
        )
        return schema(**r), usage

    def steps(self):
        return [c["step"] for c in self.calls]


PASSAGES = [
    {
        "chunk_id": "c1",
        "source_id": "en/refund-policy.md#0",
        "title": "Refund policy",
        "content": "Full refund within 14 days of first payment.",
        "rrf_score": 0.03,
        "vector_sim": 0.6,
        "fts_rank": 0.1,
    },
    {
        "chunk_id": "c2",
        "source_id": "en/invoices.md#0",
        "title": "Invoices",
        "content": "Download invoices from Billing > Invoices.",
        "rrf_score": 0.02,
        "vector_sim": 0.5,
        "fts_rank": None,
    },
]


class FakeSearch:
    def __init__(self, passages=PASSAGES):
        self.passages, self.queries = passages, []

    async def __call__(self, query, k):
        self.queries.append(query)
        return self.passages


CLS = {"category": "refund", "urgency": "normal", "language": "en"}
GOOD_DRAFT = {
    "answerable": True,
    "reply_text": "You can get a full refund within 14 days.",
    "cited_source_ids": ["en/refund-policy.md#0"],
    "confidence": 0.9,
}
GOOD_CHECK = {"supported": True, "answers_question": True, "unsupported_claims": []}


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedding: tests plumbing, not semantic quality."""

    dim = 1024

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in query_terms(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out
