from novel_kb.parsers.base_parser import BaseParser, RawDocument


class TxtParser(BaseParser):
    def parse(self, file_path: str) -> RawDocument:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        return RawDocument(title="", content=content)
