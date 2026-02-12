import re
from typing import List


def split_paragraphs(text: str, min_len: int = 60) -> List[str]:
    if not text:
        return []
    blocks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    paragraphs: List[str] = []
    for block in blocks:
        if len(block) >= min_len:
            paragraphs.append(block)
            continue
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        for line in lines:
            if len(line) >= min_len:
                paragraphs.append(line)
            elif paragraphs:
                paragraphs[-1] = f"{paragraphs[-1]} {line}".strip()
            else:
                paragraphs.append(line)
    return [p for p in paragraphs if p]
