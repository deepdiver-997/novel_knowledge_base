#!/usr/bin/env python3
"""Check consistency between chapters and metadata summaries in novel JSON.

Checks:
1) Duplicate chapter_id in chapters
2) Duplicate chapter_id in metadata.summaries.chapters
3) Missing summary entries for chapters
4) Extra summary entries not found in chapters
5) Title mismatch by chapter_id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_file(path: Path) -> int:
    data = load_json(path)

    chapters = data.get("chapters", [])
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    summaries = metadata.get("summaries", {}) if isinstance(metadata.get("summaries"), dict) else {}
    summary_chapters = summaries.get("chapters", []) if isinstance(summaries.get("chapters"), list) else []

    chapter_ids: list[str] = []
    chapter_title_map: dict[str, str] = {}
    dup_chapter_ids: set[str] = set()

    for item in chapters:
        if not isinstance(item, dict):
            continue
        chapter_id = item.get("chapter_id")
        title = item.get("title")
        if not isinstance(chapter_id, str):
            continue
        if chapter_id in chapter_title_map:
            dup_chapter_ids.add(chapter_id)
        chapter_ids.append(chapter_id)
        chapter_title_map[chapter_id] = str(title or "")

    summary_ids: list[str] = []
    summary_title_map: dict[str, str] = {}
    dup_summary_ids: set[str] = set()

    for item in summary_chapters:
        if not isinstance(item, dict):
            continue
        chapter_id = item.get("chapter_id")
        title = item.get("title")
        if not isinstance(chapter_id, str):
            continue
        if chapter_id in summary_title_map:
            dup_summary_ids.add(chapter_id)
        summary_ids.append(chapter_id)
        summary_title_map[chapter_id] = str(title or "")

    chapter_id_set = set(chapter_ids)
    summary_id_set = set(summary_ids)

    missing_summary_ids = sorted(chapter_id_set - summary_id_set)
    extra_summary_ids = sorted(summary_id_set - chapter_id_set)

    title_mismatches: list[tuple[str, str, str]] = []
    for chapter_id in sorted(chapter_id_set & summary_id_set):
        chapter_title = chapter_title_map.get(chapter_id, "")
        summary_title = summary_title_map.get(chapter_id, "")
        if chapter_title != summary_title:
            title_mismatches.append((chapter_id, chapter_title, summary_title))

    print(f"File: {path}")
    print(f"Chapters: {len(chapter_ids)}")
    print(f"Summary chapters: {len(summary_ids)}")
    print(f"Duplicate chapter_ids in chapters: {len(dup_chapter_ids)}")
    print(f"Duplicate chapter_ids in summary: {len(dup_summary_ids)}")
    print(f"Missing summary entries: {len(missing_summary_ids)}")
    print(f"Extra summary entries: {len(extra_summary_ids)}")
    print(f"Title mismatches: {len(title_mismatches)}")

    if dup_chapter_ids:
        print("\nDuplicate chapter_ids in chapters (sample):")
        for chapter_id in sorted(list(dup_chapter_ids))[:10]:
            print(f"  - {chapter_id}")

    if dup_summary_ids:
        print("\nDuplicate chapter_ids in summary (sample):")
        for chapter_id in sorted(list(dup_summary_ids))[:10]:
            print(f"  - {chapter_id}")

    if missing_summary_ids:
        print("\nMissing summary entries (sample):")
        for chapter_id in missing_summary_ids[:10]:
            print(f"  - {chapter_id}")

    if extra_summary_ids:
        print("\nExtra summary entries (sample):")
        for chapter_id in extra_summary_ids[:10]:
            print(f"  - {chapter_id}")

    if title_mismatches:
        print("\nTitle mismatches (sample):")
        for chapter_id, chapter_title, summary_title in title_mismatches[:10]:
            print(f"  - {chapter_id}")
            print(f"    chapter: {chapter_title}")
            print(f"    summary: {summary_title}")

    has_issue = any(
        [
            dup_chapter_ids,
            dup_summary_ids,
            missing_summary_ids,
            extra_summary_ids,
            title_mismatches,
        ]
    )
    return 1 if has_issue else 0


if __name__ == "__main__":
    default_path = Path.home() / ".novel_knowledge_base" / "data" / "novels" / "遮天.json"
    target = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_path
    if not target.exists():
        print(f"File not found: {target}")
        raise SystemExit(2)
    raise SystemExit(check_file(target))
