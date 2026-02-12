import re
from typing import List, Tuple

from novel_kb.segmenters.segment_model import Chapter


class ChapterSegmenter:
    @staticmethod
    def segment_epub(toc_titles: List[str], content_parts: List[str]) -> List[Chapter]:
        chapters: List[Chapter] = []
        max_len = max(len(toc_titles), len(content_parts))
        for index in range(max_len):
            title = toc_titles[index] if index < len(toc_titles) else f"Chapter {index + 1}"
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
