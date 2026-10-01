"""Paragraph-packing chunker. KB articles are short, so we keep chunks small and coherent."""


def chunk_text(body: str, max_chars: int = 700) -> list[str]:
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        pieces = [p] if len(p) <= max_chars else _split_long(p, max_chars)
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for sentence in paragraph.replace("؟", "؟\n").replace(". ", ".\n").split("\n"):
        sentence = sentence.strip()
        if not sentence:
            continue
        if cur and len(cur) + len(sentence) + 1 > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out
