import html
import re


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", text))


def normalize_whitespace(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
