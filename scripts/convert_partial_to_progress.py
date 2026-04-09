import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/convert_partial_to_progress.py <partial_json_path>")
        return 1

    partial_path = Path(sys.argv[1]).expanduser()
    if not partial_path.exists():
        print(f"File not found: {partial_path}")
        return 1

    data = json.loads(partial_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}
    progress = metadata.get("analysis_progress") or {}
    if not isinstance(progress, dict):
        print("No analysis_progress found in partial file")
        return 1

    novel_id = data.get("novel_id") or partial_path.stem.replace(".partial", "")
    payload = {
        "novel_id": novel_id,
        "title": data.get("title", ""),
        "source_path": metadata.get("source_path", ""),
        "analysis_progress": progress,
    }

    progress_path = partial_path.with_name(f"{partial_path.stem.replace('.partial', '')}.progress.json")
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
