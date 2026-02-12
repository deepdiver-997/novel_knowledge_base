from pathlib import Path

from novel_kb.config.config_schema import ParserConfig
from novel_kb.parsers.base_parser import BaseParser
from novel_kb.parsers.epub_parser import EpubParser
from novel_kb.parsers.txt_parser import TxtParser


class ParserFactory:
    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    def get_parser(self, file_path: str) -> BaseParser:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".epub":
            return EpubParser()
        if suffix == ".txt":
            return TxtParser()
        raise ValueError(f"Unsupported file type: {suffix}")
