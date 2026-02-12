from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Chapter:
    chapter_id: str
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = None
