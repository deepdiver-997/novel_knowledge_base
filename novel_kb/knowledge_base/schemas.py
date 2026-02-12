from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChapterRecord:
    chapter_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class NovelRecord:
    novel_id: str
    title: str
    summary: str
    features: List[str]
    characters: List[Dict[str, Any]]
    chapters: List[ChapterRecord]
    summary_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
