"""嵌入服务：负责生成文本向量并存入向量数据库"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.knowledge_base.schemas import NovelRecord
from novel_kb.knowledge_base.vector_store import (
    Chunk,
    ContentType,
    VectorStore,
    create_vector_store,
)
from novel_kb.utils.logger import logger
from novel_kb.utils.text import truncate_text

if TYPE_CHECKING:
    from novel_kb.llm.provider import LLMProvider


@dataclass
class EmbeddingConfig:
    """嵌入配置"""
    enabled: bool = True
    embed_summary: bool = True
    embed_chapters: bool = False  # 原文嵌入默认关闭，太大
    embed_plot_summaries: bool = True
    embed_paragraphs: bool = False  # 默认关闭，太大
    paragraph_min_chars: int = 100
    paragraph_max_chars: int = 500
    max_chars_per_embedding: int = 4000  # 嵌入时截断文本


class EmbeddingService:
    """嵌入服务"""

    def __init__(
        self,
        provider: "LLMProvider",
        vector_store: VectorStore,
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        self.provider = provider
        self.vector_store = vector_store
        self.config = config or EmbeddingConfig()

    async def embed_novel(self, record: NovelRecord) -> None:
        """为整本小说生成嵌入并存入向量库"""
        if not self.config.enabled:
            logger.info("Embedding disabled, skipping")
            return

        logger.info("Starting embedding for novel: %s", record.novel_id)
        chunks: list[Chunk] = []

        # 1. 全书总结
        if self.config.embed_summary and record.summary:
            chunk = await self._make_overall_chunk(record)
            if chunk:
                chunks.append(chunk)

        # 2. 剧情总结（卷级）
        if self.config.embed_plot_summaries:
            plot_chunks = await self._make_plot_chunks(record)
            chunks.extend(plot_chunks)

        # 3. 章节总结
        chapter_chunks = await self._make_chapter_summary_chunks(record)
        chunks.extend(chapter_chunks)

        # 4. 原文分段（可选）
        if self.config.embed_paragraphs:
            para_chunks = await self._make_paragraph_chunks(record)
            chunks.extend(para_chunks)

        # 批量存入
        if chunks:
            # 先生成所有 embedding
            chunks = await self._generate_embeddings(chunks)
            self.vector_store.upsert(chunks)
            logger.info(
                "Embedded %d chunks for novel %s (type breakdown: %s)",
                len(chunks),
                record.novel_id,
                _count_by_type(chunks),
            )

    def embed_novel_sync(self, record: NovelRecord) -> None:
        """同步版本"""
        asyncio.run(self.embed_novel(record))

    # -------------------------------------------------------------------------
    # 重新嵌入已有小说（从 JSON 读取，无需重新 LLM 分析）
    # -------------------------------------------------------------------------

    async def reindex_novel(
        self,
        repository: NovelRepository,
        novel_id: str,
        force: bool = False,
    ) -> tuple[int, int]:
        """
        重新索引单本小说的向量。

        Args:
            repository: 小说仓库
            novel_id: 小说 ID
            force: 是否强制重新嵌入（即使向量库中已有数据）

        Returns:
            (成功数量, 跳过数量)
        """
        if not self.config.enabled:
            logger.info("Embedding disabled, skipping reindex")
            return 0, 0

        # 检查向量库中是否已有数据
        existing_count = self.vector_store.count(novel_id)
        if existing_count > 0 and not force:
            logger.info(
                "Novel %s already has %d chunks in vector store, skipping (use force=True to reindex)",
                novel_id,
                existing_count,
            )
            return 0, 1

        # 删除旧数据（如果强制重新索引）
        if force and existing_count > 0:
            logger.info("Force reindex: deleting existing %d chunks for %s", existing_count, novel_id)
            self.vector_store.delete_novel(novel_id)

        # 加载小说数据
        try:
            record = repository.load_novel(novel_id)
        except Exception as exc:
            logger.error("Failed to load novel %s: %s", novel_id, exc)
            return 0, 0

        # 生成嵌入
        await self.embed_novel(record)

        # 统计成功数量
        new_count = self.vector_store.count(novel_id)
        return new_count, 0

    def reindex_novel_sync(
        self,
        repository: NovelRepository,
        novel_id: str,
        force: bool = False,
    ) -> tuple[int, int]:
        """同步版本"""
        return asyncio.run(self.reindex_novel(repository, novel_id, force))

    async def reindex_all_novels(
        self,
        repository: NovelRepository,
        force: bool = False,
        novel_ids: Optional[list[str]] = None,
    ) -> dict[str, tuple[int, int]]:
        """
        批量重新索引所有小说。

        Args:
            repository: 小说仓库
            force: 是否强制重新嵌入
            novel_ids: 指定要索引的小说 ID 列表（None = 全部）

        Returns:
            {novel_id: (成功数量, 跳过数量)}
        """
        if not self.config.enabled:
            logger.info("Embedding disabled, skipping reindex")
            return {}

        # 获取要处理的小说列表
        if novel_ids is None:
            novels = repository.list_novels_metadata()
            novel_ids = [n.novel_id for n in novels]
            logger.info("Found %d novels to check for reindexing", len(novel_ids))
        else:
            logger.info("Reindexing %d specified novels", len(novel_ids))

        results: dict[str, tuple[int, int]] = {}

        for novel_id in novel_ids:
            try:
                success, skipped = await self.reindex_novel(repository, novel_id, force)
                results[novel_id] = (success, skipped)
                if skipped > 0:
                    logger.info("Skipped %s (already indexed)", novel_id)
                else:
                    logger.info("Reindexed %s: %d chunks", novel_id, success)
            except Exception as exc:
                logger.error("Failed to reindex %s: %s", novel_id, exc)
                results[novel_id] = (0, 0)

        # 汇总
        total_success = sum(r[0] for r in results.values())
        total_skipped = sum(r[1] for r in results.values())
        logger.info(
            "Reindex complete: %d novels, %d chunks indexed, %d skipped",
            len(results),
            total_success,
            total_skipped,
        )
        return results

    def reindex_all_novels_sync(
        self,
        repository: NovelRepository,
        force: bool = False,
        novel_ids: Optional[list[str]] = None,
    ) -> dict[str, tuple[int, int]]:
        """批量重新索引（同步版本）"""
        return asyncio.run(self.reindex_all_novels(repository, force, novel_ids))

    async def _make_overall_chunk(self, record: NovelRecord) -> Optional[Chunk]:
        """生成全书总结 chunk"""
        if not record.summary:
            return None
        text = truncate_text(record.summary, max_chars=self.config.max_chars_per_embedding)
        return Chunk(
            chunk_id=f"{record.novel_id}/overall/overall",
            novel_id=record.novel_id,
            novel_title=record.title,
            content=f"【全书总结】{text}",
            content_type=ContentType.OVERALL_SUMMARY,
        )

    async def _make_plot_chunks(self, record: NovelRecord) -> list[Chunk]:
        """生成剧情总结 chunks"""
        chunks: list[Chunk] = []
        summaries = record.metadata.get("summaries", {})
        plots = summaries.get("plots", []) if isinstance(summaries, dict) else []

        for i, plot in enumerate(plots):
            if not isinstance(plot, dict):
                continue
            summary = plot.get("summary", "")
            if not summary:
                continue
            text = truncate_text(summary, max_chars=self.config.max_chars_per_embedding)
            volume = plot.get("volume_title", "未分卷")
            start = plot.get("start_chapter", "")
            end = plot.get("end_chapter", "")
            content = f"【{volume}】{start} - {end}\n{text}"

            chunks.append(Chunk(
                chunk_id=f"{record.novel_id}/plot/v_{i}",
                novel_id=record.novel_id,
                novel_title=record.title,
                content=content,
                content_type=ContentType.PLOT_SUMMARY,
                volume_title=volume,
            ))
        return chunks

    async def _make_chapter_summary_chunks(self, record: NovelRecord) -> list[Chunk]:
        """生成章节总结 chunks"""
        chunks: list[Chunk] = []
        summaries = record.metadata.get("summaries", {})
        chapter_summaries = summaries.get("chapters", []) if isinstance(summaries, dict) else []
        summary_map = {str(item.get("chapter_id", "")): item for item in chapter_summaries}

        for i, chapter in enumerate(record.chapters):
            summary_item = summary_map.get(chapter.chapter_id, {})
            summary_text = summary_item.get("summary", "") if isinstance(summary_item, dict) else ""
            if not summary_text:
                continue
            text = truncate_text(summary_text, max_chars=self.config.max_chars_per_embedding)
            content = f"【{chapter.title}】{text}"

            # 尝试获取 volume_title
            volume_title = self._detect_volume(chapter.title)

            chunks.append(Chunk(
                chunk_id=f"{record.novel_id}/chapter_summary/{chapter.chapter_id}",
                novel_id=record.novel_id,
                novel_title=record.title,
                content=content,
                content_type=ContentType.CHAPTER_SUMMARY,
                chapter_id=chapter.chapter_id,
                chapter_index=i,
                chapter_title=chapter.title,
                volume_title=volume_title,
            ))
        return chunks

    async def _make_paragraph_chunks(self, record: NovelRecord) -> list[Chunk]:
        """生成原文段落 chunks（按章节分段）"""
        chunks: list[Chunk] = []
        min_chars = self.config.paragraph_min_chars
        max_chars = self.config.paragraph_max_chars

        for i, chapter in enumerate(record.chapters):
            if not chapter.content:
                continue
            paragraphs = _split_paragraphs(chapter.content, min_len=min_chars)

            for j, para in enumerate(paragraphs):
                # 截断超长段落
                para = truncate_text(para, max_chars=max_chars)
                volume_title = self._detect_volume(chapter.title)

                chunks.append(Chunk(
                    chunk_id=f"{record.novel_id}/paragraph/{chapter.chapter_id}_p{j}",
                    novel_id=record.novel_id,
                    novel_title=record.title,
                    content=para,
                    content_type=ContentType.PARAGRAPH,
                    chapter_id=chapter.chapter_id,
                    chapter_index=i,
                    chapter_title=chapter.title,
                    volume_title=volume_title,
                ))
        return chunks

    async def _generate_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """批量生成 embedding"""
        semaphore = asyncio.Semaphore(5)  # 并发控制

        async def embed_one(chunk: Chunk) -> Chunk:
            async with semaphore:
                try:
                    result = await self.provider.generate_embedding(chunk.content)
                    if result and result.vector:
                        chunk.embedding = result.vector
                except Exception as exc:
                    logger.warning("Embedding failed for chunk %s: %s", chunk.chunk_id, exc)
                return chunk

        return await asyncio.gather(*[embed_one(c) for c in chunks])

    @staticmethod
    def _detect_volume(title: str) -> Optional[str]:
        """从章节标题检测卷名"""
        if not title:
            return None
        match = re.search(r"第.{1,6}卷", title)
        if match:
            return match.group(0)
        match = re.search(r"Volume\s+\w+", title, re.IGNORECASE)
        if match:
            return match.group(0)
        if "卷" in title and len(title) <= 20:
            return title.strip()
        return None


def _split_paragraphs(text: str, min_len: int = 100) -> list[str]:
    """按段落分割文本"""
    # 先按换行分割
    lines = text.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append("\n".join(current))
                current = []
            continue
        current.append(stripped)
        if len("\n".join(current)) >= min_len:
            paragraphs.append("\n".join(current))
            current = []

    if current:
        paragraphs.append("\n".join(current))

    return [p for p in paragraphs if p.strip()]


def _count_by_type(chunks: list[Chunk]) -> str:
    """统计各类型数量"""
    counts: dict[str, int] = {}
    for c in chunks:
        ct = c.content_type.value if isinstance(c.content_type, ContentType) else str(c.content_type)
        counts[ct] = counts.get(ct, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in counts.items())