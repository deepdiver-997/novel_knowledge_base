import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from novel_kb.knowledge_base.schemas import ChapterRecord, NovelMetadata, NovelRecord


class NovelRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.novels_dir = self.base_dir / "novels"
        self.novels_dir.mkdir(parents=True, exist_ok=True)

    def save_novel(self, record: NovelRecord) -> None:
        path = self._record_path(record.novel_id)
        data = asdict(record)
        self._write_json_atomic(path, data)

    def save_progress(self, novel_id: str, payload: Dict) -> None:
        path = self._progress_path(novel_id)
        self._write_json_atomic(path, payload)

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

    def load_progress(self, novel_id: str) -> Dict:
        path = self._progress_path(novel_id)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_novel_ids(self) -> List[str]:
        return [path.stem for path in self.novels_dir.glob("*.json")]

    def list_novels(self) -> List[NovelRecord]:
        return [self.load_novel(novel_id) for novel_id in self.list_novel_ids()]

    def list_novels_metadata(self) -> List[NovelMetadata]:
        """仅加载轻量级元数据，不加载完整内容"""
        results: List[NovelMetadata] = []
        for path in self.novels_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                results.append(NovelMetadata(
                    novel_id=data.get("novel_id", path.stem),
                    title=data.get("title", ""),
                    created_at=data.get("created_at", ""),
                ))
            except Exception:
                continue
        return results

    def exists(self, novel_id: str) -> bool:
        return self._record_path(novel_id).exists()

    def progress_exists(self, novel_id: str) -> bool:
        return self._progress_path(novel_id).exists()

    def delete_progress(self, novel_id: str) -> None:
        path = self._progress_path(novel_id)
        if path.exists():
            path.unlink()

    def _record_path(self, novel_id: str) -> Path:
        safe_id = novel_id.replace("/", "_")
        return self.novels_dir / f"{safe_id}.json"

    def _progress_path(self, novel_id: str) -> Path:
        safe_id = novel_id.replace("/", "_")
        return self.novels_dir / f"{safe_id}.progress.json"

    @staticmethod
    def _write_json_atomic(path: Path, data: Dict) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
