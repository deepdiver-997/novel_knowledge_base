from novel_kb.parsers.base_parser import BaseParser, RawDocument
from novel_kb.utils.text import normalize_whitespace, strip_html


class EpubParser(BaseParser):
    def parse(self, file_path: str) -> RawDocument:
        try:
            from ebooklib import epub, ITEM_DOCUMENT
        except ImportError as exc:
            raise RuntimeError("ebooklib is required for EPUB parsing") from exc

        book = epub.read_epub(file_path)
        title = book.get_metadata("DC", "title")
        title_text = title[0][0] if title else ""
        items = [item for item in book.get_items_of_type(ITEM_DOCUMENT)]
        parts = []
        for item in items:
            raw = item.get_content().decode("utf-8", errors="ignore")
            parts.append(normalize_whitespace(strip_html(raw)))
        content = "\n".join(parts)
        toc = [getattr(t, "title", "") for t in book.toc if getattr(t, "title", None)]
        return RawDocument(title=title_text, content=content, toc=toc, parts=parts)
