import asyncio
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.knowledge_base.schemas import ChapterRecord, NovelRecord
from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.llm.provider import LLMProvider
from novel_kb.parsers.factory import ParserFactory
from novel_kb.segmenters.chapter_segmenter import ChapterSegmenter
from novel_kb.utils.segment import split_paragraphs
from novel_kb.utils.text import truncate_text
from novel_kb.utils.logger import logger


@dataclass
class AnalysisOptions:
    segment_enabled: bool = True
    segment_min_chars: int = 120
    segment_max_chars: Optional[int] = None
    concurrency_limit: int = 2
    qps_limit: float = 1.0
    retry_limit: int = 3
    retry_interval: float = 1.0


class _AsyncRateLimiter:
    """异步 QPS 速率限制"""
    def __init__(self, qps: float) -> None:
        self.qps = qps
        self._next_time = 0.0

    async def wait(self) -> None:
        if not self.qps or self.qps <= 0:
            return
        import time
        interval = 1.0 / self.qps
        now = time.monotonic()
        if now < self._next_time:
            sleep_for = self._next_time - now
            if sleep_for >= 0.5:
                logger.info("Rate limiting: sleeping %.2fs", sleep_for)
            await asyncio.sleep(sleep_for)
        self._next_time = max(now, self._next_time) + interval


class IngestService:
    """异步版本的摄入服务"""

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        provider: Optional[LLMProvider],
        repository: NovelRepository,
    ) -> None:
        self.config = config
        self.provider = provider
        self.repository = repository
        self.parser_factory = ParserFactory(config.parser)
        self.embedding_builder = None
        if config.storage.embedding_enabled and provider:
            self.embedding_builder = EmbeddingBuilder(provider)

    def ingest_file(
        self,
        file_path: str,
        novel_id: Optional[str] = None,
        title: Optional[str] = None,
        overwrite: bool = False,
        analysis_options: Optional[AnalysisOptions] = None,
    ) -> NovelRecord:
        """同步入口 - 运行异步任务"""
        return asyncio.run(
            self.ingest_file_async(
                file_path,
                novel_id=novel_id,
                title=title,
                overwrite=overwrite,
                analysis_options=analysis_options,
            )
        )

    def resume_file(
        self,
        file_path: str,
        novel_id: Optional[str] = None,
        title: Optional[str] = None,
        analysis_options: Optional[AnalysisOptions] = None,
    ) -> NovelRecord:
        """同步入口 - 运行异步任务"""
        return asyncio.run(
            self.resume_file_async(
                file_path,
                novel_id=novel_id,
                title=title,
                analysis_options=analysis_options,
            )
        )

    async def ingest_file_async(
        self,
        file_path: str,
        novel_id: Optional[str] = None,
        title: Optional[str] = None,
        overwrite: bool = False,
        analysis_options: Optional[AnalysisOptions] = None,
    ) -> NovelRecord:
        """异步摄入文件"""
        return await self._ingest(
            file_path,
            novel_id=novel_id,
            title=title,
            overwrite=overwrite,
            analysis_options=analysis_options,
            resume=False,
        )

    async def resume_file_async(
        self,
        file_path: str,
        novel_id: Optional[str] = None,
        title: Optional[str] = None,
        analysis_options: Optional[AnalysisOptions] = None,
    ) -> NovelRecord:
        """异步恢复摄入"""
        return await self._ingest(
            file_path,
            novel_id=novel_id,
            title=title,
            overwrite=False,
            analysis_options=analysis_options,
            resume=True,
        )

    async def _ingest(
        self,
        file_path: str,
        novel_id: Optional[str],
        title: Optional[str],
        overwrite: bool,
        analysis_options: Optional[AnalysisOptions],
        resume: bool,
    ) -> NovelRecord:
        """核心异步摄入逻辑"""
        logger.info("Ingest start: file=%s resume=%s", file_path, resume)
        parser = self.parser_factory.get_parser(file_path)
        document = parser.parse(file_path)

        resolved_title = title or document.title or Path(file_path).stem
        resolved_id = novel_id or self._slugify(resolved_title)

        if resume and self.repository.exists(resolved_id) and not self.repository.progress_exists(resolved_id):
            return self.repository.load_novel(resolved_id)

        if not overwrite and not resume and self.repository.exists(resolved_id):
            return self.repository.load_novel(resolved_id)

        chapters = self._segment_document(document)

        progress = self._load_progress(resolved_id) if resume else None
        try:
            summary, features, characters, progress_state = await self._analyze_document(
                chapters,
                analysis_options,
                progress,
                resume=resume,
                resolved_id=resolved_id,
                resolved_title=resolved_title,
                source_path=str(Path(file_path).expanduser()),
            )
        except Exception as exc:
            logger.error("Ingest failed during analysis: %s", exc)
            raise

        if not summary:
            logger.error("Ingest failed: empty summary")
            raise RuntimeError("Summary generation failed")

        summary_embedding, chapter_embeddings = await self._build_embeddings(summary, chapters)

        record = NovelRecord(
            novel_id=resolved_id,
            title=resolved_title,
            summary=summary,
            features=features,
            characters=characters,
            chapters=chapters,
            summary_embedding=summary_embedding,
            metadata={
                "source_path": str(Path(file_path).expanduser()),
                **self._final_metadata(progress_state),
            },
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
        )
        self.repository.save_novel(record)
        if self.repository.progress_exists(resolved_id):
            self.repository.delete_progress(resolved_id)
        logger.info("Ingest complete: novel_id=%s", resolved_id)
        return record

    def _load_progress(self, novel_id: str) -> Optional[dict]:
        if not self.repository.progress_exists(novel_id):
            return None
        data = self.repository.load_progress(novel_id)
        progress = data.get("analysis_progress")
        return progress if isinstance(progress, dict) else None

    @staticmethod
    def _final_metadata(progress_state: Optional[dict]) -> dict:
        if not isinstance(progress_state, dict):
            return {}
        summaries = progress_state.get("summaries")
        if isinstance(summaries, dict):
            return {"summaries": summaries}
        return {}

    @staticmethod
    def _normalize_progress(progress: Optional[dict]) -> dict:
        if not isinstance(progress, dict):
            return {"chapter_index": 0, "chapter_summaries": [], "characters": []}
        chapter_index = progress.get("chapter_index", 0)
        summaries = progress.get("chapter_summaries", [])
        characters = progress.get("characters", [])
        return {
            "chapter_index": int(chapter_index) if isinstance(chapter_index, int) else 0,
            "chapter_summaries": summaries if isinstance(summaries, list) else [],
            "characters": characters if isinstance(characters, list) else [],
        }

    def _segment_document(self, document) -> list[ChapterRecord]:
        if document.parts:
            toc_titles = document.toc or []
            chapters = ChapterSegmenter.segment_epub(toc_titles, document.parts)
        else:
            chapters = ChapterSegmenter.segment_txt(
                document.content,
                chapter_pattern=self.config.parser.txt_chapter_pattern,
            )
        records = []
        for index, chapter in enumerate(chapters):
            metadata = chapter.metadata or {}
            metadata.setdefault("chapter_index", index)
            metadata.setdefault("chapter_title", chapter.title)
            records.append(
                ChapterRecord(
                    chapter_id=chapter.chapter_id,
                    title=chapter.title,
                    content=chapter.content,
                    metadata=metadata,
                )
            )
        return records

    async def _analyze_document(
        self,
        chapters: list[ChapterRecord],
        analysis_options: Optional[AnalysisOptions],
        progress: Optional[dict],
        resume: bool,
        resolved_id: str,
        resolved_title: str,
        source_path: str,
    ) -> tuple[str, list[str], list[dict], Optional[dict]]:
        if not self.provider or not self.config.storage.analysis_enabled:
            return "", [], [], None

        return await self._analyze_hierarchical(
            chapters,
            analysis_options,
            progress,
            resume,
            resolved_id,
            resolved_title,
            source_path,
        )

    async def _analyze_hierarchical(
        self,
        chapters: list[ChapterRecord],
        analysis_options: Optional[AnalysisOptions],
        progress: Optional[dict],
        resume: bool,
        resolved_id: str,
        resolved_title: str,
        source_path: str,
    ) -> tuple[str, list[str], list[dict], Optional[dict]]:
        """分层分析：先分段汇总章节，再按卷/分组汇总，最后总览"""
        options = self._resolve_analysis_options(analysis_options)
        normalized = self._normalize_progress(progress) if resume else {
            "chapter_index": 0,
            "chapter_summaries": [],
            "characters": [],
        }
        progress_state = None
        start_index = normalized.get("chapter_index", 0)
        chapter_summaries = list(normalized.get("chapter_summaries", []))
        all_characters = list(normalized.get("characters", []))

        try:
            summary_max_tokens = self._summary_max_tokens()
            logger.info(
                "Hierarchical summary: chapters=%d start_index=%d",
                len(chapters),
                start_index,
            )
            for index, chapter in enumerate(chapters):
                if index < start_index:
                    continue
                logger.info("Chapter summary start: %s (index=%d)", chapter.title, index)
                chapter_content = truncate_text(chapter.content, max_chars=self.config.storage.analysis_max_chars)

                if not chapter_content.strip():
                    summary_status = "chapter_content_empty"
                    chapter_summaries.append(self._build_chapter_summary_item(chapter, "", summary_status))
                    self._write_partial_progress(
                        resolved_id,
                        resolved_title,
                        source_path,
                        chapters,
                        index + 1,
                        chapter_summaries,
                        all_characters,
                    )
                    logger.warning(
                        "Chapter content empty, skipping analysis: %s (index=%d)",
                        chapter.title,
                        index,
                    )
                    continue

                chapter_summary = await self._summarize_text(
                    chapter_content,
                    options,
                    max_tokens=None,
                )
                summary_status = self._chapter_summary_status(chapter_content, chapter_summary)
                if chapter_summary:
                    chapter_summaries.append(self._build_chapter_summary_item(chapter, chapter_summary, summary_status))
                    self._write_partial_progress(
                        resolved_id,
                        resolved_title,
                        source_path,
                        chapters,
                        index + 1,
                        chapter_summaries,
                        all_characters,
                    )
                    logger.info("Chapter summary done: %s (index=%d)", chapter.title, index)
                else:
                    chapter_summaries.append(self._build_chapter_summary_item(chapter, "", summary_status))
                    self._write_partial_progress(
                        resolved_id,
                        resolved_title,
                        source_path,
                        chapters,
                        index + 1,
                        chapter_summaries,
                        all_characters,
                    )
                    logger.error(
                        "Chapter summary failed: %s (index=%d) reason=%s",
                        chapter.title,
                        index,
                        summary_status,
                    )
                    raise RuntimeError("Chapter summary failed")

                if self.config.storage.characters_enabled:
                    logger.info("Character extraction start: %s (index=%d)", chapter.title, index)
                    char_result = await self._extract_characters_segments(chapter_content, options)
                    if isinstance(char_result, dict) and "characters" in char_result:
                        all_characters.extend(char_result["characters"])
                    elif isinstance(char_result, list):
                        all_characters.extend(char_result)
                    else:
                        logger.warning(
                            "Empty character result for chapter: %s (index=%s)",
                            chapter.title,
                            index,
                        )
                    self._write_partial_progress(
                        resolved_id,
                        resolved_title,
                        source_path,
                        chapters,
                        index + 1,
                        chapter_summaries,
                        all_characters,
                    )
                    logger.info("Character extraction done: %s (index=%d)", chapter.title, index)

            if not chapter_summaries:
                return "", [], [], progress_state

            logger.info("Plot summaries start: volumes")
            plot_summaries = await self._summarize_by_volume(
                chapters,
                chapter_summaries,
                options,
                summary_max_tokens,
            )
            logger.info("Plot summaries done: count=%d", len(plot_summaries))

            overall_source = [item["summary"] for item in plot_summaries] or [
                item["summary"] for item in chapter_summaries if isinstance(item, dict)
            ]
            logger.info("Overall summary start")
            final_summary = await self._summarize_items(overall_source, options, summary_max_tokens)
            if not final_summary:
                logger.error("Overall summary failed")
                raise RuntimeError("Overall summary failed")
            logger.info("Overall summary done: length=%d", len(final_summary))

            features = []
            if self.config.storage.features_enabled and chapter_summaries:
                feature_result = (await self.provider.extract_features(
                    truncate_text(
                        "\n".join(overall_source),
                        max_chars=self.config.storage.analysis_max_chars,
                    )
                )).content
                if isinstance(feature_result, dict) and "features" in feature_result:
                    features = feature_result["features"]

            return (
                final_summary or "",
                features if isinstance(features, list) else [],
                self._dedupe_characters(all_characters),
                {
                    **(progress_state or {}),
                    "summaries": {
                        "chapters": chapter_summaries,
                        "plots": plot_summaries,
                        "overall": final_summary or "",
                    },
                },
            )
        except Exception as exc:
            logger.error("Hierarchical analysis failed: %s", exc)
            raise

    def _write_partial_progress(
        self,
        resolved_id: str,
        resolved_title: str,
        source_path: str,
        chapters: list[ChapterRecord],
        chapter_index: int,
        chapter_summaries: list[dict],
        characters: list[dict],
    ) -> None:
        progress_state = {
            "analysis_progress": {
                "chapter_index": chapter_index,
                "chapter_summaries": chapter_summaries,
                "characters": characters,
            }
        }
        payload = {
            "novel_id": resolved_id,
            "title": resolved_title,
            "source_path": source_path,
            **progress_state,
        }
        self.repository.save_progress(resolved_id, payload)

    def _resolve_analysis_options(self, analysis_options: Optional[AnalysisOptions]) -> AnalysisOptions:
        if analysis_options is not None:
            return analysis_options
        segment_max_chars = self.config.storage.segment_max_chars
        return AnalysisOptions(
            segment_enabled=True,
            segment_min_chars=self.config.storage.segment_min_chars,
            segment_max_chars=segment_max_chars or None,
            concurrency_limit=self.config.storage.segment_concurrency,
            qps_limit=self.config.storage.segment_qps,
            retry_limit=self.config.storage.segment_retries,
            retry_interval=self.config.storage.segment_retry_interval,
        )

    @staticmethod
    def _chapter_summary_status(chapter_content: str, summary: str, had_error: bool = False) -> str:
        if not str(chapter_content or "").strip():
            return "chapter_content_empty"
        if had_error:
            return "model_summary_error"
        if not str(summary or "").strip():
            return "model_summary_empty"
        return "ok"

    @staticmethod
    def _build_chapter_summary_item(chapter: ChapterRecord, summary: str, summary_status: str) -> dict[str, Any]:
        return {
            "chapter_id": chapter.chapter_id,
            "title": chapter.title,
            "summary": summary,
            "summary_status": summary_status,
        }

    async def _summarize_text(
        self,
        text: str,
        options: AnalysisOptions,
        max_tokens: Optional[int],
    ) -> str:
        segments = self._segment_text(text, options)
        if not segments:
            return ""
        lengths = [len(segment) for segment in segments]
        logger.info(
            "Summarize text: segments=%d min=%d max=%d avg=%.1f",
            len(segments),
            min(lengths),
            max(lengths),
            sum(lengths) / len(lengths),
        )
        summaries = await self._map_with_limits(
            segments,
            options,
            lambda segment: self.provider.analyze_plot(segment),
        )
        cleaned = [self._extract_summary(item) for item in summaries if item]
        combined = "\n".join(part for part in cleaned if part)
        return await self._summarize_until_fit(combined, options, max_tokens)

    async def _summarize_items(
        self,
        items: list[str],
        options: AnalysisOptions,
        max_tokens: Optional[int],
    ) -> str:
        if not items:
            return ""
        chunks = self._chunk_items_by_chars(items, self.config.storage.analysis_max_chars)
        lengths = [len(chunk) for chunk in chunks] or [0]
        logger.info(
            "Summarize items: items=%d chunks=%d min=%d max=%d avg=%.1f",
            len(items),
            len(chunks),
            min(lengths),
            max(lengths),
            sum(lengths) / len(lengths),
        )
        summaries = await self._map_with_limits(
            chunks,
            options,
            lambda chunk: self.provider.analyze_plot(chunk),
        )
        cleaned = [self._extract_summary(item) for item in summaries if item]
        combined = "\n".join(part for part in cleaned if part)
        return await self._summarize_until_fit(combined, options, max_tokens)

    async def _summarize_until_fit(
        self,
        combined: str,
        options: AnalysisOptions,
        max_tokens: Optional[int],
    ) -> str:
        max_chars = self.config.storage.analysis_max_chars
        attempts = 0
        while combined and len(combined) > max_chars and attempts < 3:
            logger.info(
                "Summarize until fit: pass=%d length=%d max_chars=%d",
                attempts + 1,
                len(combined),
                max_chars,
            )
            segments = self._segment_text(combined, options)
            summaries = await self._map_with_limits(
                segments,
                options,
                lambda segment: self.provider.analyze_plot(segment),
            )
            cleaned = [self._extract_summary(item) for item in summaries if item]
            new_combined = "\n".join(part for part in cleaned if part)
            if not new_combined or len(new_combined) >= len(combined):
                break
            combined = new_combined
            attempts += 1
        combined = truncate_text(combined, max_chars=max_chars)
        result = await self._analyze_plot_with_max_tokens(combined, max_tokens)
        return self._extract_summary(result)

    async def _summarize_by_volume(
        self,
        chapters: list[ChapterRecord],
        chapter_summaries: list[dict],
        options: AnalysisOptions,
        max_tokens: Optional[int],
    ) -> list[dict]:
        summary_map = {
            item.get("chapter_id"): item for item in chapter_summaries if isinstance(item, dict)
        }
        groups: list[dict] = []
        current_volume = "未分卷"
        current_items: list[str] = []
        current_start: Optional[ChapterRecord] = None
        current_end: Optional[ChapterRecord] = None

        for chapter in chapters:
            volume_title = self._detect_volume_title(chapter.title)
            if volume_title:
                if current_items:
                    groups.append({
                        "volume_title": current_volume,
                        "start_chapter": current_start.title if current_start else "",
                        "end_chapter": current_end.title if current_end else "",
                        "summaries": list(current_items),
                    })
                    current_items = []
                current_volume = volume_title
                current_start = None
                current_end = None

            summary_item = summary_map.get(chapter.chapter_id)
            if not summary_item:
                continue
            summary_text = summary_item.get("summary")
            if not summary_text:
                continue
            current_items.append(f"{chapter.title}: {summary_text}")
            if current_start is None:
                current_start = chapter
            current_end = chapter

        if current_items:
            groups.append({
                "volume_title": current_volume,
                "start_chapter": current_start.title if current_start else "",
                "end_chapter": current_end.title if current_end else "",
                "summaries": list(current_items),
            })

        plot_summaries: list[dict] = []
        for group in groups:
            summary = await self._summarize_items(group["summaries"], options, max_tokens)
            plot_summaries.append({
                "volume_title": group["volume_title"],
                "start_chapter": group["start_chapter"],
                "end_chapter": group["end_chapter"],
                "summary": summary,
            })
        return plot_summaries

    @staticmethod
    def _chunk_items_by_chars(items: list[str], max_chars: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for item in items:
            if not item:
                continue
            if len(item) > max_chars:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_len = 0
                for index in range(0, len(item), max_chars):
                    chunks.append(item[index:index + max_chars])
                continue
            if current_len + len(item) + 1 <= max_chars:
                current.append(item)
                current_len += len(item) + 1
            else:
                if current:
                    chunks.append("\n".join(current))
                current = [item]
                current_len = len(item)
        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def _detect_volume_title(title: str) -> Optional[str]:
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

    async def _analyze_plot_with_max_tokens(self, text: str, max_tokens: Optional[int]) -> dict[str, Any] | str:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            logger.warning("Skip analyze_plot: empty text after preprocessing")
            return {"summary": ""}
        if not max_tokens or max_tokens <= 0:
            result = await self.provider.analyze_plot(normalized_text)
            return result.content
        original = getattr(self.provider, "max_tokens", None)
        if original is None:
            result = await self.provider.analyze_plot(normalized_text)
            return result.content
        self.provider.max_tokens = max_tokens
        try:
            result = await self.provider.analyze_plot(normalized_text)
            return result.content
        finally:
            self.provider.max_tokens = original

    def _summary_max_tokens(self) -> Optional[int]:
        value = getattr(self.config.llm, "summary_max_tokens", None)
        if isinstance(value, int) and value > 0:
            return value
        return None

    async def _extract_characters_segments(self, text: str, options: AnalysisOptions) -> list[dict]:
        segments = self._segment_text(text, options)
        if not segments:
            return []
        results = await self._map_with_limits(
            segments,
            options,
            lambda segment: self.provider.analyze_characters(segment),
        )
        characters: list[dict] = []
        for item in results:
            if isinstance(item, dict) and "characters" in item:
                characters.extend(item["characters"])
            elif isinstance(item, list):
                characters.extend(item)
        return self._dedupe_characters(characters)

    def _segment_text(self, text: str, options: AnalysisOptions) -> list[str]:
        max_chars = options.segment_max_chars or self.config.storage.analysis_max_chars
        min_chars = options.segment_min_chars
        paragraphs = split_paragraphs(text, min_len=min_chars)
        segments: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            if not paragraph:
                continue
            if len(paragraph) > max_chars:
                if current:
                    segments.append("\n".join(current))
                    current = []
                    current_len = 0
                for index in range(0, len(paragraph), max_chars):
                    segments.append(paragraph[index:index + max_chars])
                continue
            if current_len + len(paragraph) + 1 <= max_chars:
                current.append(paragraph)
                current_len += len(paragraph) + 1
            else:
                if current:
                    segments.append("\n".join(current))
                current = [paragraph]
                current_len = len(paragraph)

        if current:
            segments.append("\n".join(current))

        if not segments:
            segments = [truncate_text(text, max_chars=max_chars)] if text else []
        return segments

    async def _map_with_limits(
        self,
        items: Iterable[str],
        options: AnalysisOptions,
        func: Callable[[str], Any],
    ) -> list[Any]:
        """异步并发执行，支持速率限制和重试"""
        if self.provider and hasattr(self.provider, "configure_limits"):
            try:
                self.provider.configure_limits(options.concurrency_limit, options.qps_limit)
            except Exception as exc:
                logger.warning("Failed to configure provider limits: %s", exc)
        limiter = _AsyncRateLimiter(options.qps_limit)
        concurrency = max(1, options.concurrency_limit)

        indexed_items = list(enumerate(items))
        total = len(indexed_items)
        logger.info(
            "Map with limits: total=%d concurrency=%d qps=%.2f retries=%d",
            total,
            concurrency,
            options.qps_limit,
            options.retry_limit,
        )

        # 使用信号量限制并发
        semaphore = asyncio.Semaphore(concurrency)

        async def task(item: str, index: int) -> Any:
            """单个任务，支持重试"""
            import time
            async with semaphore:
                normalized_item = str(item or "").strip()
                if not normalized_item:
                    logger.warning(
                        "Segment skipped: %d/%d reason=empty_input",
                        index + 1,
                        total,
                    )
                    return {"summary": "", "summary_status": "empty_input"}
                for attempt in range(options.retry_limit + 1):
                    start_time = time.monotonic()
                    try:
                        await limiter.wait()
                        # func 返回 AnalysisResult，我们需要 .content
                        result = await func(normalized_item)
                        elapsed = time.monotonic() - start_time
                        logger.info(
                            "Segment done: %d/%d attempt=%d elapsed=%.2fs",
                            index + 1,
                            total,
                            attempt + 1,
                            elapsed,
                        )
                        # 提取 content 字段
                        return result.content if hasattr(result, 'content') else result
                    except Exception as exc:
                        elapsed = time.monotonic() - start_time
                        logger.warning(
                            "Segment failed: %d/%d attempt=%d elapsed=%.2fs error=%s",
                            index + 1,
                            total,
                            attempt + 1,
                            elapsed,
                            exc,
                        )
                        if attempt >= options.retry_limit:
                            raise
                        await asyncio.sleep(options.retry_interval)

        # 创建所有任务
        tasks = [task(item, index) for index, item in indexed_items]
        # 并发执行，保持顺序
        results = await asyncio.gather(*tasks)
        return results

    @staticmethod
    def _extract_summary(item: Any) -> str:
        if isinstance(item, dict):
            for key in ("summary", "answer", "content", "raw"):
                value = item.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        return text
                elif isinstance(value, dict):
                    nested = value.get("summary") or value.get("answer") or value.get("content")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
                elif isinstance(value, list):
                    joined = "\n".join(str(v).strip() for v in value if str(v).strip())
                    if joined:
                        return joined
                else:
                    text = str(value).strip()
                    if text:
                        return text
            return ""
        if item is None:
            return ""
        return str(item)

    @staticmethod
    def _dedupe_characters(characters: list[dict]) -> list[dict]:
        seen = set()
        result: list[dict] = []
        for character in characters:
            name = str(character.get("name", "")).strip().lower() if isinstance(character, dict) else ""
            key = name or json.dumps(character, ensure_ascii=False, sort_keys=True) if isinstance(character, dict) else str(character)
            if key in seen:
                continue
            seen.add(key)
            if isinstance(character, dict):
                result.append(character)
        return result

    async def _build_embeddings(
        self,
        summary: str,
        chapters: list[ChapterRecord],
    ) -> tuple[Optional[list[float]], dict[str, list[float]]]:
        if not self.embedding_builder:
            return None, {}

        summary_embedding = None
        if self.config.storage.embed_summary and summary:
            embedding_text = truncate_text(summary, max_chars=self.config.storage.embedding_max_chars)
            summary_embedding = (await self.embedding_builder.build_async(embedding_text)).vector

        chapter_embeddings: dict[str, list[float]] = {}
        if self.config.storage.embed_chapters:
            tasks = []
            for chapter in chapters:
                embedding_text = truncate_text(chapter.content, max_chars=self.config.storage.embedding_max_chars)
                tasks.append(self.embedding_builder.build_async(embedding_text))
            results = await asyncio.gather(*tasks)
            for chapter, result in zip(chapters, results):
                chapter_embeddings[chapter.chapter_id] = result.vector

        if chapter_embeddings:
            for chapter in chapters:
                if chapter.chapter_id in chapter_embeddings:
                    chapter.embedding = chapter_embeddings[chapter.chapter_id]

        return summary_embedding, chapter_embeddings

    @staticmethod
    def _fallback_summary(chapters: list[ChapterRecord]) -> str:
        if not chapters:
            return ""
        return truncate_text(chapters[0].content, max_chars=800)

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
        return "-".join(part for part in cleaned.split("-") if part)
