# SupportPilot evaluation: `dev` split

| Run | |
|---|---|
| split | dev |
| timestamp_utc | 20261007-121937 |
| git_commit | unknown (no git in this environment) |
| generator_model | openai/gpt-oss-20b |
| judge_model | openai/gpt-oss-20b |
| embedding_model | bge-m3 |
| retrieval_k | 5 |
| min_confidence | 0.6 |
| prompt_version | v1 |
| prompts_sha | d60844793d |
| dataset_sha | 89c5d7d762 |
| llm_calls_from_cache | 0/67 |

Tickets: 20 (0 pipeline errors). Intervals are Wilson 95% and are wide at this sample size: treat differences of a few points as noise.

## Summary

| Metric | Result | n |
|---|---|---|
| Retrieval hit@1 | 92.9%  (95% CI 69–99%) | 14 |
| Retrieval hit@3 | 92.9%  (95% CI 69–99%) | 14 |
| Retrieval hit@5 | 100.0%  (95% CI 78–100%) | 14 |
| Correct abstention (KB has no answer → human) | 100.0%  (95% CI 61–100%) | 6 |
| False abstention (KB has the answer, still → human) | 7.1%  (95% CI 1–31%) | 14 |
| Answer coverage (answerable tickets that got a draft) | 92.9%  (95% CI 69–99%) | 14 |
| Faithfulness (drafts fully supported, LLM judge) | 100.0%  (95% CI 76–100%) | 12 |
| Claim-level support (LLM judge) | 100.0%  (95% CI 93–100%) | 51 |
| Fact recall, among drafted answers | 95.5%  (95% CI 78–99%) | 22 |
| Fact recall, end-to-end (abstaining = 0) | 80.8%  (95% CI 62–91%) | 26 |
| Injection resistance (no forbidden text in final draft) | 100.0%  (95% CI 44–100%) | 3 |
| Classification: category accuracy | 85.0%  (95% CI 64–95%) | 20 |
| Classification: urgency accuracy | 65.0%  (95% CI 43–82%) | 20 |
| Classification: language accuracy | 95.0%  (95% CI 76–99%) | 20 |
| MRR (retrieval) | 0.946 | 14 |
| Human: approved as-is | 100.0%  (95% CI 77–100%) | 13 |
| Human: edited | 0.0%  (95% CI 0–23%) | 13 |
| Human: rejected | 0.0%  (95% CI 0–23%) | 13 |
| **Human-edit rate** (edited / accepted) | 0.0%  (95% CI 0–23%) | 13 |

## By language

| lang | answerable | hit@3 | answered | unanswerable | correct abstention |
|---|---|---|---|---|---|
| en | 8 | 7 | 8 | 3 | 3 |
| fa | 5 | 5 | 4 | 3 | 3 |
| mixed | 1 | 1 | 1 | 0 | 0 |

## Cost and latency

- Agent latency per ticket: p50 15636 ms, p95 27923 ms (excludes judge; retrieval is memoised, so embedding time is not included)
- Tokens: 40380 in / 9511 out
- Estimated cost: agent $0.00254 + judge $0.00148 (from `LLM_PRICE_*`; includes calls served from cache at their original price)
- Judge: 1 failed, 0 drafts had no factual claims (excluded from faithfulness)

## Diagnostics

- Median top-1 vector similarity: answerable **0.700** vs unanswerable **0.477**. If these are close, a similarity threshold cannot separate 'KB has the answer' from 'KB does not', which is why abstention relies on the drafter's `answerable` flag, deterministic guards and the fact-check instead.
- Routed to a human, by reason: {'llm_error': 1, 'kb_has_no_answer': 6}

## Human review

- Reviewed drafts: 13.
- Spot check of the judge: 0 drafts labelled by hand; human says faithful n/a; judge agrees with the human on n/a.

## Error analysis (auto-generated)

### Pipeline crashes (0)

_None._

### Retrieval misses (gold article not in top-5) (0)

_None._

### False abstentions (answerable, but routed to a human) (1)

| id | lang | reason | discarded draft |
|---|---|---|---|
| d04 | fa | llm_error | اگر پرداخت شما ناموفق بود، ابتدا مطمئن شوید که موجودی کارت کافی است، کارت منقضی نشده و برای پرداخت اینترنتی و… |

### Answered although the KB has no answer (0)

_None._

### Unfaithful drafts (judge found unsupported claims) (0)

_None._

### Missing gold facts in answered drafts (1)

| id | missing facts |
|---|---|
| d01 | full refund within 14 days of the first payment |

### Injection leaks (0)

_None._

### Classification errors (11)

| id | field | gold | predicted |
|---|---|---|---|
| d07 | urgency | low | normal |
| d09 | urgency | low | normal |
| d11 | language | mixed | fa |
| d13 | urgency | low | normal |
| d14 | category | technical | other |
| d15 | category | billing | other |
| d16 | urgency | low | normal |
| d17 | category | other | technical |
| d17 | urgency | low | normal |
| d18 | urgency | low | normal |
| d19 | urgency | low | normal |

## Author's analysis

_Written by hand after reading the failure tables above: what failed, why, what was changed on `dev`, and what was deliberately left unfixed._
