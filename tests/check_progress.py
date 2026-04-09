#!/usr/bin/env python3
"""检查progress文件中的章节内容"""
import json
import sys

def check_progress_chapter(progress_path: str, chapter_index: int):
    with open(progress_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    progress = data.get('analysis_progress', {})
    chapter_summaries = progress.get('chapter_summaries', [])
    
    print(f"已处理章节数: {progress.get('chapter_index', 0)}")
    print(f"已有摘要数: {len(chapter_summaries)}")
    
    # 查找指定章节
    target_summary = None
    for summary in chapter_summaries:
        if summary.get('chapter_id') == chapter_index:
            target_summary = summary
            break
    
    if target_summary:
        print(f"\n=== 找到章节 {chapter_index} 的摘要 ===")
        print(f"标题: {target_summary.get('title')}")
        print(f"摘要长度: {len(target_summary.get('summary', ''))}")
        print(f"摘要内容: {target_summary.get('summary', '')[:500]}")
    else:
        print(f"\n未找到章节 {chapter_index} 的摘要")
        print(f"\n最后几个章节的信息:")
        for summary in chapter_summaries[-5:]:
            print(f"  - chapter_id={summary.get('chapter_id')}, title={summary.get('title')}, summary_len={len(summary.get('summary', ''))}")

if __name__ == "__main__":
    progress_path = "/Users/zhuhongrui/.novel_knowledge_base/data/novels/遮天.progress.json"
    chapter_index = 999
    
    if len(sys.argv) > 1:
        chapter_index = int(sys.argv[1])
    
    check_progress_chapter(progress_path, chapter_index)
