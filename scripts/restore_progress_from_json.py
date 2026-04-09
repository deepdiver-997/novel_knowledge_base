import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/restore_progress_from_json.py <novel_json_path>")
        return 1

    novel_path = Path(sys.argv[1]).expanduser()
    if not novel_path.exists():
        print(f"File not found: {novel_path}")
        return 1

    progress_path = novel_path.with_name(f"{novel_path.stem}.progress.json")

    data = json.loads(novel_path.read_text(encoding="utf-8"))
    chapters = data.get("chapters", []) if isinstance(data.get("chapters"), list) else []
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    summaries = metadata.get("summaries", {}) if isinstance(metadata.get("summaries"), dict) else {}
    chapter_summaries = summaries.get("chapters", []) if isinstance(summaries.get("chapters"), list) else []

    summary_map = {}
    for item in chapter_summaries:
        if not isinstance(item, dict):
            continue
        chapter_id = item.get("chapter_id")
        if isinstance(chapter_id, str):
            summary_map[chapter_id] = {
                "chapter_id": chapter_id,
                "title": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
            }

    ordered = []
    contiguous_idx = 0
    for chapter in chapters:
        if not isinstance(chapter, dict):
            break
        chapter_id = chapter.get("chapter_id")
        if not isinstance(chapter_id, str):
            break
        item = summary_map.get(chapter_id)
        if not item:
            break
        ordered.append(item)
        if str(item.get("summary", "")).strip():
            contiguous_idx += 1
        else:
            break

    payload = {
        "novel_id": data.get("novel_id", novel_path.stem),
        "title": data.get("title", novel_path.stem),
        "source_path": metadata.get("source_path", ""),
        "analysis_progress": {
            "chapter_index": contiguous_idx,
            "chapter_summaries": ordered,
            "characters": data.get("characters", []) if isinstance(data.get("characters"), list) else [],
        },
    }

    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {progress_path}")
    print(f"chapter_index={contiguous_idx}, chapter_summaries={len(ordered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
