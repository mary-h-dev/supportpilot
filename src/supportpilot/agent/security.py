"""Ticket text is UNTRUSTED. Defenses here + in graph.py:

1. The LLM never chooses or calls tools: it only returns JSON. Code calls the allowlisted tools.
2. Untrusted text is HTML-escaped (can't close our delimiter tag) and placed in a labelled block.
3. System prompts say: block content is data, never instructions.
4. Output is validated: cited ids must come from retrieval; claims are fact-checked vs passages.
5. Suspicious phrasing is flagged for the human reviewer (flag only: false positives are common).
"""

import html
import re

MAX_TICKET_CHARS = 4000

_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above|your)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now\b",
    r"\bsystem\s*:",
    r"\bact as\b",
    r"developer mode|jailbreak",
    r"دستور(ات|الع?مل(‌|\s)?های?)? (قبلی|بالا)",
    r"(قبلی|بالا)\s*را\s*(نادیده|فراموش)",
    r"(نادیده|فراموش)\s*بگیر",
    r"پرامپت|دستورات سیستم|پیام سیستم",
]
_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def detect_injection(text: str) -> list[str]:
    return ["possible_prompt_injection"] if _RE.search(text) else []


def sanitize_untrusted(text: str) -> str:
    """Truncate + escape < > & so the text cannot break out of its delimiter tag."""
    return html.escape(text[:MAX_TICKET_CHARS], quote=False)


def ticket_block(subject: str, body: str, sender: str = "") -> str:
    return (
        "<ticket_untrusted>\n"
        f"<subject>{sanitize_untrusted(subject)}</subject>\n"
        f"<body>{sanitize_untrusted(body)}</body>\n"
        "</ticket_untrusted>"
    )
