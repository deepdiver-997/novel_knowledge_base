#!/usr/bin/env python3
"""Sync progress chapter titles from a novel JSON file.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/fix_progress_titles_from_novel.py \
    /path/to/novel.json /path/to/novel.progress.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def build_title_map(novel_data: dict) -> dict[str, str]:
    chapters = novel_data.get("chapters", [])
    title_map: dict[str, str] = {}
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_id = chapter.get("chapter_id")
        title = chapter.get("title")
        if isinstance(chapter_id, str) and isinstance(title, str):
            title_map[chapter_id] = title
    return title_map


def fix_progress(novel_path: Path, progress_path: Path) -> None:
    novel_data = load_json(novel_path)
    progress_data = load_json(progress_path)

    title_map = build_title_map(novel_data)
    progress = progress_data.get("analysis_progress", {})
    chapter_summaries = progress.get("chapter_summaries", [])
    if not isinstance(chapter_summaries, list):
        print("No chapter_summaries list found in progress file.")
        return

    updated = 0
    missing = 0
    for summary in chapter_summaries:
        if not isinstance(summary, dict):
            continue
        chapter_id = summary.get("chapter_id")
        if not isinstance(chapter_id, str):
            continue
        new_title = title_map.get(chapter_id)
        if not new_title:
            missing += 1
            continue
        old_title = summary.get("title")
        if old_title != new_title:
            summary["title"] = new_title
            updated += 1

    save_json(progress_path, progress_data)
    print(
        f"Updated {updated} chapter summary titles. "
        f"Missing title mappings: {missing}."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: fix_progress_titles_from_novel.py <novel.json> <novel.progress.json>")
        sys.exit(1)

    novel_path = Path(sys.argv[1]).expanduser()
    progress_path = Path(sys.argv[2]).expanduser()

    if not novel_path.exists():
        print(f"Novel file not found: {novel_path}")
        sys.exit(1)
    if not progress_path.exists():
        print(f"Progress file not found: {progress_path}")
        sys.exit(1)

    fix_progress(novel_path, progress_path)
