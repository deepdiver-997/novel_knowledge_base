#!/usr/bin/env python3
"""Fix chapter titles in existing JSON by extracting headings from content.

This does not rerun any LLM calls. It only updates chapter titles and metadata
based on the content's leading heading.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHAPTER_PREFIX_RE = re.compile(
    r"^第[〇零一二三四五六七八九十百千万两0-9]+章$",
    re.UNICODE,
)


def extract_heading(text: str) -> str | None:
    if not text:
        return None
    sample = text[:200].strip()
    if not sample:
        return None
    parts = re.split(r"\s+", sample)
    if not parts:
        return None
    if not CHAPTER_PREFIX_RE.match(parts[0]):
        return None
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def fix_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chapters = data.get("chapters", [])
    if not isinstance(chapters, list):
        print("No chapters list found.")
        return

    updated = 0
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        content = chapter.get("content", "") or ""
        heading = extract_heading(content)
        if not heading:
            continue
        current_title = str(chapter.get("title", "") or "")
        if current_title != heading:
            chapter["title"] = heading
            metadata = chapter.get("metadata")
            if isinstance(metadata, dict):
                metadata["chapter_title"] = heading
            updated += 1

    title_map: dict[str, str] = {}
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_id = chapter.get("chapter_id")
        title = chapter.get("title")
        if isinstance(chapter_id, str) and isinstance(title, str):
            title_map[chapter_id] = title

    summaries = data.get("metadata", {}).get("summaries", {})
    summaries_chapters = summaries.get("chapters", []) if isinstance(summaries, dict) else []
    summary_updated = 0
    if isinstance(summaries_chapters, list) and title_map:
        for item in summaries_chapters:
            if not isinstance(item, dict):
                continue
            chapter_id = item.get("chapter_id")
            if not isinstance(chapter_id, str):
                continue
            new_title = title_map.get(chapter_id)
            if new_title and item.get("title") != new_title:
                item["title"] = new_title
                summary_updated += 1

    if updated == 0 and summary_updated == 0:
        print("No chapter titles updated.")
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        f"Updated {updated} chapter titles and {summary_updated} summary titles in {path}."
    )


if __name__ == "__main__":
    default_path = Path.home() / ".novel_knowledge_base" / "data" / "novels" / "遮天.json"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    fix_file(target)
