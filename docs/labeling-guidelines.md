# Labeling guidelines (gold labels for the evaluation)

These definitions are the single source of truth: the same wording is used in the classifier
prompt (`agent/prompts.py`) and when labelling `eval/tickets.json`. They were written down after
the first dev run showed that the original labels were applied inconsistently at the boundaries.
Every label was made by one person (the author): see "Limitations" in the README.

## Category
| category | covers |
|---|---|
| billing | payments and payment methods, charges, invoices, prices, discounts, plan or seat costs |
| refund | the customer wants money back |
| cancellation | cancelling a subscription or deleting an account |
| account_access | login, password, two-factor, email or identity changes |
| technical | how the product works, bugs and errors, usage limits, integrations, platform or feature availability |
| other | anything else: company, security / data-hosting / compliance / legal, sales or support-channel questions |

## Urgency
| urgency | meaning |
|---|---|
| high | money was taken wrongly or lost, the customer is locked out, or data is lost |
| normal | the customer has a concrete problem or request but is not blocked |
| low | informational question with no problem to solve (how-to, availability, pricing, policy) |

**Known weakness:** the low / normal boundary is subjective, and the classifier did not learn it: in the v2 dev run it predicted `normal` for all 11 gold-`low` tickets, which is exactly the accuracy of "always normal except high" (47.6%). The longer rubric above is the *labelling* rule; the classifier prompt uses the shorter v1 wording because the longer one measured no better (see `PROMPT_VERSION` v3). The operationally important decision
is *escalating `high`*, so the report leads with high-urgency recall and precision and shows
3-level accuracy only as a secondary number.

## Language
`en`, `fa`, or `mixed`. `mixed` only if both languages appear in substantial parts (full phrases
or sentences). A Persian message with a few English technical terms is `fa`; an English message
with a few Persian words is `en`. (For replies this barely matters: the draft is written in the
language of the customer's main question.)

## Answerable
A ticket is *answerable* only if the knowledge base states what the customer needs. If the KB
only touches the topic (e.g. says "contact support" without the asked-for detail), it is *not*
answerable, and the correct behaviour is to hand it to a human.

## Change log
- **v2 (after the first dev run, before any test run):** definitions made explicit; `mixed`
  clarified, so `d11` and `t06` (Persian with English technical terms) were relabelled `fa`; `d21`
  (a genuinely mixed ticket) added to dev so both splits keep a true `mixed` case; `t13`
  (double charge, high urgency, not in the KB) added to test so the held-out set has a high-urgency
  case. No test result had been seen when these changes were made.
- **v3 (after the clean v2 dev run, before any test run):** billing names payment methods and
  other names data-hosting questions (generic clarifications, no ticket text copied into prompts).
  Urgency prompt wording reverted to v1 (see above). Dev is frozen after this.
