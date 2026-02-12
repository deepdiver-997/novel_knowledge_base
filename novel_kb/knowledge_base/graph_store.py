import json
from pathlib import Path
from typing import Dict, List


class GraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def upsert_characters(self, novel_id: str, characters: List[Dict[str, str]]) -> None:
        data = self._read()
        data[novel_id] = {"characters": characters}
        self._write(data)

    def get_characters(self, novel_id: str) -> Dict[str, List[Dict[str, str]]]:
        data = self._read()
        return data.get(novel_id, {"characters": []})

    def _read(self) -> Dict[str, Dict]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: Dict[str, Dict]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
