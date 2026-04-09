#!/usr/bin/env python3
"""检查特定章节的内容"""
import sys
from novel_kb.parsers.epub_parser import EpubParser
from novel_kb.segmenters.chapter_segmenter import ChapterSegmenter

def check_chapter(epub_path: str, chapter_index: int):
    # 解析epub
    parser = EpubParser()
    raw_doc = parser.parse(epub_path)
    
    print(f"书名: {raw_doc.title}")
    print(f"总内容长度: {len(raw_doc.content)} 字符")
    
    # 使用segmenter切分章节
    chapters = ChapterSegmenter.segment_epub(raw_doc.toc or [], raw_doc.parts or [])
    
    print(f"总章节数: {len(chapters)}")
    
    if chapter_index >= len(chapters):
        print(f"章节索引 {chapter_index} 超出范围")
        return
    
    chapter = chapters[chapter_index]
    print(f"\n=== 章节 {chapter_index} ===")
    print(f"章节ID: {chapter.chapter_id}")
    print(f"标题: {chapter.title}")
    print(f"内容长度: {len(chapter.content)} 字符")
    print(f"内容为空: {not chapter.content.strip()}")
    
    # 显示前500字符
    if chapter.content:
        print(f"\n前500字符:")
        print(chapter.content[:500])
        print("\n...")
        print(f"\n后500字符:")
        print(chapter.content[-500:])
    else:
        print("\n内容为空!")

if __name__ == "__main__":
    epub_path = "/Users/zhuhongrui/Downloads/apps/SoNovel-macOS_arm64/downloads/遮天(辰东).epub"
    chapter_index = 999
    
    if len(sys.argv) > 1:
        chapter_index = int(sys.argv[1])
    
    check_chapter(epub_path, chapter_index)
