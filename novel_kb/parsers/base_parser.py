from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RawDocument:
    title: str
    content: str
    toc: Optional[List[str]] = None
    parts: Optional[List[str]] = None


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> RawDocument:
        raise NotImplementedError
