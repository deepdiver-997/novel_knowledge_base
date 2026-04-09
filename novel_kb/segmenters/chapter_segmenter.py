import re
from typing import List, Tuple

from novel_kb.segmenters.segment_model import Chapter


class ChapterSegmenter:
    @staticmethod
    def _normalize_epub_titles(toc_titles: List[str], content_parts: List[str]) -> List[str]:
        titles = [title.strip() for title in toc_titles if title]
        if not titles:
            return titles

        cover_titles = {"封面", "目录", "扉页", "前言", "序", "序章", "引子", "楔子"}
        first = titles[0]
        if first in cover_titles:
            if len(titles) == len(content_parts) + 1:
                return titles[1:]
            if content_parts:
                sample = content_parts[0][:200]
                if first not in sample:
                    if len(titles) > 1 and titles[1] in sample:
                        return titles[1:]
                    if len(titles) > len(content_parts):
                        return titles[1:]
        return titles

    @staticmethod
    def segment_epub(toc_titles: List[str], content_parts: List[str]) -> List[Chapter]:
        chapters: List[Chapter] = []
        titles = ChapterSegmenter._normalize_epub_titles(toc_titles, content_parts)
        max_len = max(len(titles), len(content_parts))
        for index in range(max_len):
            title = titles[index] if index < len(titles) else f"Chapter {index + 1}"
            content = content_parts[index] if index < len(content_parts) else ""
            chapters.append(
                Chapter(
                    chapter_id=f"ch_{index:04d}",
                    title=title,
                    content=content,
                    metadata={"source": "epub"},
                )
            )
        return chapters

    @staticmethod
    def segment_txt(content: str, chapter_pattern: str) -> List[Chapter]:
        pattern = re.compile(chapter_pattern, re.MULTILINE)
        matches = list(pattern.finditer(content))
        if not matches:
            return [Chapter(chapter_id="ch_0000", title="", content=content.strip())]
        chapters: List[Chapter] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            title = match.group(0)
            chapters.append(
                Chapter(
                    chapter_id=f"ch_{index:04d}",
                    title=title.strip(),
                    content=content[start:end].strip(),
                    metadata={"source": "txt"},
                )
            )
        return chapters
