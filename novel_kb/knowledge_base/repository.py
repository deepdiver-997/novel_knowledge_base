import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from novel_kb.knowledge_base.schemas import ChapterRecord, NovelRecord


class NovelRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.novels_dir = self.base_dir / "novels"
        self.novels_dir.mkdir(parents=True, exist_ok=True)

    def save_novel(self, record: NovelRecord) -> None:
        path = self._record_path(record.novel_id)
        data = asdict(record)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def load_novel(self, novel_id: str) -> NovelRecord:
        path = self._record_path(novel_id)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        chapters = [ChapterRecord(**item) for item in data.get("chapters", [])]
        return NovelRecord(
            novel_id=data["novel_id"],
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            features=data.get("features", []),
            characters=data.get("characters", []),
            chapters=chapters,
            summary_embedding=data.get("summary_embedding"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
        )

    def list_novel_ids(self) -> List[str]:
        return [path.stem for path in self.novels_dir.glob("*.json")]

    def list_novels(self) -> List[NovelRecord]:
        return [self.load_novel(novel_id) for novel_id in self.list_novel_ids()]

    def exists(self, novel_id: str) -> bool:
        return self._record_path(novel_id).exists()

    def _record_path(self, novel_id: str) -> Path:
        safe_id = novel_id.replace("/", "_")
        return self.novels_dir / f"{safe_id}.json"
