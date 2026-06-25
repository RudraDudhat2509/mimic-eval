from __future__ import annotations

import re

# split after . ! ? (optionally followed by quotes/brackets) when followed by whitespace
_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+')


def split(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _BOUNDARY.split(text)]
    parts = [p for p in parts if p]
    return parts or [text]
