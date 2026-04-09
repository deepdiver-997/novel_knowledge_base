import asyncio
import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.knowledge_base.schemas import ChapterRecord, NovelRecord
from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.llm.provider import LLMProvider
from novel_kb.parsers.factory import ParserFactory
from novel_kb.services.summary_audit import find_suspicious_summaries, load_fingerprints
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
    strict_mode: bool = False
    chapter_max_tokens: Optional[int] = None
    strict_retry_interval: float = 2.0
    fingerprint_path: Optional[str] = None
    audit_min_score: float = 0.9
    audit_similarity_threshold: float = 0.72
    audit_min_length: int = 20


ChapterTaskStatus = Literal[
    "PENDING",
    "SEGMENTS_READY",
    "RUNNING",
    "REDUCE_READY",
    "REDUCING",
    "DONE",
    "FAILED",
]


@dataclass
class ChapterTaskState:
    chapter_id: str
    chapter_index: int
    title: str
    status: ChapterTaskStatus = "PENDING"
    segment_texts: list[str] = field(default_factory=list)
    segment_results: dict[int, str] = field(default_factory=dict)
    segment_errors: dict[int, str] = field(default_factory=dict)
    chapter_summary: str = ""
    attempts_segment: dict[int, int] = field(default_factory=dict)
    attempts_reduce: int = 0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.updated_at:
            import time
            self.updated_at = time.monotonic()

    def touch(self) -> None:
        import time
        self.updated_at = time.monotonic()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "chapter_index": self.chapter_index,
            "title": self.title,
            "status": self.status,
            "segment_texts": self.segment_texts,
            "segment_results": {str(k): v for k, v in self.segment_results.items()},
            "segment_errors": {str(k): v for k, v in self.segment_errors.items()},
            "chapter_summary": self.chapter_summary,
            "attempts_segment": {str(k): v for k, v in self.attempts_segment.items()},
            "attempts_reduce": self.attempts_reduce,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ChapterTaskState":
        segment_results_raw = data.get("segment_results", {})
        segment_errors_raw = data.get("segment_errors", {})
        attempts_segment_raw = data.get("attempts_segment", {})
        return ChapterTaskState(
            chapter_id=str(data.get("chapter_id", "")),
            chapter_index=int(data.get("chapter_index", 0)),
            title=str(data.get("title", "")),
            status=data.get("status", "PENDING"),
            segment_texts=data.get("segment_texts", []) if isinstance(data.get("segment_texts"), list) else [],
            segment_results={
                int(k): str(v) for k, v in segment_results_raw.items()
            } if isinstance(segment_results_raw, dict) else {},
            segment_errors={
                int(k): str(v) for k, v in segment_errors_raw.items()
            } if isinstance(segment_errors_raw, dict) else {},
            chapter_summary=str(data.get("chapter_summary", "")),
            attempts_segment={
                int(k): int(v) for k, v in attempts_segment_raw.items()
            } if isinstance(attempts_segment_raw, dict) else {},
            attempts_reduce=int(data.get("attempts_reduce", 0)),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
        )


@dataclass
class WorkStateMap:
    chapters: dict[str, ChapterTaskState]
    inflight_segment_tasks: int = 0
    inflight_reduce_tasks: int = 0
    order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapters": {chapter_id: state.to_dict() for chapter_id, state in self.chapters.items()},
            "inflight_segment_tasks": self.inflight_segment_tasks,
            "inflight_reduce_tasks": self.inflight_reduce_tasks,
            "order": self.order,
        }

    @staticmethod
    def from_dict(data: dict[str, Any], chapters: list[ChapterRecord]) -> "WorkStateMap":
        chapter_states: dict[str, ChapterTaskState] = {}
        raw_chapters = data.get("chapters", {}) if isinstance(data.get("chapters"), dict) else {}
        for chapter_id, raw_state in raw_chapters.items():
            if not isinstance(chapter_id, str) or not isinstance(raw_state, dict):
                continue
            chapter_states[chapter_id] = ChapterTaskState.from_dict(raw_state)

        for index, chapter in enumerate(chapters):
            if chapter.chapter_id not in chapter_states:
                chapter_states[chapter.chapter_id] = ChapterTaskState(
                    chapter_id=chapter.chapter_id,
                    chapter_index=index,
                    title=chapter.title,
                )

        raw_order = data.get("order", []) if isinstance(data.get("order"), list) else []
        order = [item for item in raw_order if isinstance(item, str)]
        if not order:
            order = [chapter.chapter_id for chapter in sorted(chapter_states.values(), key=lambda item: item.chapter_index)]

        return WorkStateMap(
            chapters=chapter_states,
            inflight_segment_tasks=int(data.get("inflight_segment_tasks", 0) or 0),
            inflight_reduce_tasks=int(data.get("inflight_reduce_tasks", 0) or 0),
            order=order,
        )


@dataclass
class RuntimeTaskRegistry:
    segment_futures: dict[asyncio.Task[Any], tuple[str, int]]
    reduce_futures: dict[asyncio.Task[Any], str]
    segment_slots: dict[tuple[str, int], asyncio.Task[Any]]
    reduce_slots: dict[str, asyncio.Task[Any]]
    created_at: float
    last_tick_at: float

    @staticmethod
    def create() -> "RuntimeTaskRegistry":
        import time
        now = time.monotonic()
        return RuntimeTaskRegistry(
            segment_futures={},
            reduce_futures={},
            segment_slots={},
            reduce_slots={},
            created_at=now,
            last_tick_at=now,
        )


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

    INVALID_SUMMARY_PATTERNS = [
        r"请提供.*(内容|主题|文本)",
        r"未提供.*(主题|内容|文本)",
        r"无法生成.*(总结|摘要)",
        r"我是一个ai助手",
        r"这是一个(示例|通用|简短).*(总结|摘要)",
        r"json格式的输出",
        r"return\s+json",
        r"rate\s*limit|429|too\s*many\s*requests|api\s*error|providererror",
    ]

    @staticmethod
    def _compact_error_text(error: Any, max_chars: int = 240) -> str:
        text = str(error or "")
        lowered = text.lower()
        if "<!doctype html" in lowered or "<html" in lowered:
            if "502" in lowered or "bad gateway" in lowered:
                return "upstream_502_bad_gateway_html"
            return "upstream_html_error_response"
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 1] + "…"

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

    def _sanitize_summary_text(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        for pattern in self.INVALID_SUMMARY_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                logger.warning("Discard suspicious summary output by pattern: %s", pattern)
                return ""
        return normalized

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
        effective_options = self._resolve_analysis_options(analysis_options)

        progress = self._load_progress(resolved_id) if resume else None
        try:
            summary, features, characters, progress_state = await self._analyze_document(
                chapters,
                effective_options,
                progress,
                resume=resume,
                resolved_id=resolved_id,
                resolved_title=resolved_title,
                source_path=str(Path(file_path).expanduser()),
            )
        except Exception as exc:
            logger.error("Ingest failed during analysis: %s", exc)
            raise

        if not effective_options.strict_mode:
            logger.info(
                "Normal mode enabled: keep progress only, skip final summary publish. "
                "Use --strict-mode to run final aggregation and persist final novel file."
            )
            return NovelRecord(
                novel_id=resolved_id,
                title=resolved_title,
                summary="",
                features=features,
                characters=characters,
                chapters=chapters,
                summary_embedding=None,
                metadata={
                    "source_path": str(Path(file_path).expanduser()),
                    "status": "progress_only",
                    **self._final_metadata(progress_state),
                },
                created_at=datetime.datetime.utcnow().isoformat() + "Z",
            )

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
            return {
                "chapter_index": 0,
                "chapter_summaries": [],
                "characters": [],
            }
        chapter_index = progress.get("chapter_index", 0)
        summaries = progress.get("chapter_summaries", [])
        characters = progress.get("characters", [])
        return {
            "chapter_index": int(chapter_index) if isinstance(chapter_index, int) else 0,
            "chapter_summaries": summaries if isinstance(summaries, list) else [],
            "characters": characters if isinstance(characters, list) else [],
        }

    @staticmethod
    def _build_work_state_map(chapters: list[ChapterRecord], work_map_data: Optional[dict]) -> WorkStateMap:
        if isinstance(work_map_data, dict):
            return WorkStateMap.from_dict(work_map_data, chapters)
        chapter_states = {
            chapter.chapter_id: ChapterTaskState(
                chapter_id=chapter.chapter_id,
                chapter_index=index,
                title=chapter.title,
            )
            for index, chapter in enumerate(chapters)
        }
        return WorkStateMap(
            chapters=chapter_states,
            order=[chapter.chapter_id for chapter in chapters],
        )

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
        work_state_map = self._build_work_state_map(chapters, None)
        _runtime_registry = RuntimeTaskRegistry.create()

        try:
            summary_max_tokens = self._summary_max_tokens()
            logger.info(
                "Hierarchical summary: chapters=%d start_index=%d",
                len(chapters),
                start_index,
            )
            chapter_summaries, all_characters = await self._summarize_chapters_windowed(
                chapters,
                options,
                start_index,
                chapter_summaries,
                all_characters,
                resolved_id,
                resolved_title,
                source_path,
                work_state_map,
                _runtime_registry,
            )

            if not chapter_summaries:
                return "", [], [], progress_state

            if options.strict_mode:
                fingerprint_file = options.fingerprint_path or str(Path.cwd() / "wrong_fliter.txt")
                fingerprints = load_fingerprints(Path(fingerprint_file).expanduser())
                if fingerprints:
                    while True:
                        suspicious = find_suspicious_summaries(
                            summary_items=chapter_summaries,
                            fingerprints=fingerprints,
                            min_length=max(1, int(options.audit_min_length)),
                            similarity_threshold=max(0.0, min(1.0, float(options.audit_similarity_threshold))),
                        )
                        suspicious = [item for item in suspicious if item.score >= max(0.0, float(options.audit_min_score))]
                        if not suspicious:
                            break
                        retry_ids = {item.chapter_id for item in suspicious if item.chapter_id}
                        sample = ", ".join(sorted(list(retry_ids))[:5])
                        logger.warning(
                            "Strict mode audit found %d suspicious chapter summaries; regenerating. sample=%s",
                            len(retry_ids),
                            sample or "-",
                        )
                        chapter_summaries, all_characters = await self._summarize_chapters_windowed(
                            chapters,
                            options,
                            start_index=0,
                            chapter_summaries=chapter_summaries,
                            all_characters=all_characters,
                            resolved_id=resolved_id,
                            resolved_title=resolved_title,
                            source_path=source_path,
                            work_state_map=work_state_map,
                            runtime_registry=_runtime_registry,
                            force_retry_ids=retry_ids,
                        )
                else:
                    logger.warning("Strict mode audit fingerprint file not found or empty: %s", fingerprint_file)
            else:
                logger.info("Normal mode: skip plot/overall final summary generation")
                return (
                    "",
                    [],
                    self._dedupe_characters(all_characters),
                    {
                        **(progress_state or {}),
                        "summaries": {
                            "chapters": chapter_summaries,
                            "plots": [],
                            "overall": "",
                        },
                    },
                )

            non_empty_ids = {
                str(item.get("chapter_id", ""))
                for item in chapter_summaries
                if isinstance(item, dict) and str(item.get("summary", "")).strip()
            }
            missing_or_empty = [
                chapter
                for chapter in chapters
                if chapter.chapter_id not in non_empty_ids
            ]
            if missing_or_empty:
                examples = ", ".join(
                    f"{item.title}(index={index})"
                    for index, item in enumerate(missing_or_empty[:5])
                )
                raise RuntimeError(
                    "Chapter summaries incomplete: "
                    f"{len(missing_or_empty)} chapters are empty or missing. "
                    f"Examples: {examples}"
                )

            logger.info("Plot summaries start: volumes")
            plot_summaries = await self._summarize_by_volume(
                chapters,
                chapter_summaries,
                options,
                summary_max_tokens,
            )
            logger.info("Plot summaries done: count=%d", len(plot_summaries))

            overall_source = [
                item["summary"]
                for item in plot_summaries
                if isinstance(item, dict) and str(item.get("summary", "")).strip()
            ] or [
                item["summary"]
                for item in chapter_summaries
                if isinstance(item, dict) and str(item.get("summary", "")).strip()
            ]
            logger.info("Overall summary start")
            if overall_source:
                final_summary = await self._summarize_items(overall_source, options, summary_max_tokens)
            else:
                final_summary = ""
            if not final_summary:
                logger.warning("Overall summary empty, using fallback summary")
                final_summary = "全书有效章节摘要较少，已跳过空章节并完成分层处理。"
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

    @staticmethod
    def _ordered_chapter_summaries(chapters: list[ChapterRecord], summary_by_id: dict[str, dict]) -> list[dict]:
        ordered: list[dict] = []
        for chapter in chapters:
            item = summary_by_id.get(chapter.chapter_id)
            if isinstance(item, dict):
                ordered.append(item)
        return ordered

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

    @staticmethod
    def _contiguous_prefix_summaries(chapters: list[ChapterRecord], summary_by_id: dict[str, dict]) -> list[dict]:
        ordered: list[dict] = []
        for chapter in chapters:
            item = summary_by_id.get(chapter.chapter_id)
            if not isinstance(item, dict):
                break
            ordered.append(item)
        return ordered

    @staticmethod
    def _contiguous_done_index(chapters: list[ChapterRecord], summary_by_id: dict[str, dict]) -> int:
        index = 0
        for chapter in chapters:
            if chapter.chapter_id not in summary_by_id:
                break
            index += 1
        return index

    async def _summarize_chapters_windowed(
        self,
        chapters: list[ChapterRecord],
        options: AnalysisOptions,
        start_index: int,
        chapter_summaries: list[dict],
        all_characters: list[dict],
        resolved_id: str,
        resolved_title: str,
        source_path: str,
        work_state_map: WorkStateMap,
        runtime_registry: RuntimeTaskRegistry,
        force_retry_ids: Optional[set[str]] = None,
    ) -> tuple[list[dict], list[dict]]:
        force_retry_ids = force_retry_ids or set()
        summary_by_id: dict[str, dict] = {}
        effective_summary_ids: set[str] = set()
        for item in chapter_summaries:
            if isinstance(item, dict):
                chapter_id = item.get("chapter_id")
                if isinstance(chapter_id, str) and chapter_id:
                    summary_by_id[chapter_id] = item
                    summary_text = str(item.get("summary", ""))
                    if summary_text.strip() and chapter_id not in force_retry_ids:
                        effective_summary_ids.add(chapter_id)

        for index, chapter in enumerate(chapters):
            if (
                index < start_index
                and chapter.chapter_id in effective_summary_ids
                and chapter.chapter_id not in force_retry_ids
            ):
                state = work_state_map.chapters.get(chapter.chapter_id)
                if state:
                    state.status = "DONE"
                    state.chapter_summary = str(summary_by_id[chapter.chapter_id].get("summary", ""))
                    state.touch()

        pending_indices = [
            index
            for index, chapter in enumerate(chapters)
            if chapter.chapter_id not in effective_summary_ids
        ]

        if not pending_indices:
            return self._ordered_chapter_summaries(chapters, summary_by_id), all_characters

        chapter_window = max(1, min(3, options.concurrency_limit))
        active: dict[asyncio.Task[Any], int] = {}
        next_pending = 0

        async def _run_one(index: int, chapter: ChapterRecord) -> tuple[str, list[dict], bool, str]:
            logger.info("Chapter summary start: %s (index=%d)", chapter.title, index)
            chapter_state = work_state_map.chapters.get(chapter.chapter_id)
            if chapter_state:
                chapter_state.status = "RUNNING"
                chapter_state.touch()

            if not chapter.content or not chapter.content.strip():
                logger.warning(
                    "Chapter %s (index=%d) has empty content, skipping analysis",
                    chapter.title,
                    index,
                )
                return "", [], False, "chapter_content_empty"

            chapter_content = truncate_text(chapter.content, max_chars=self.config.storage.analysis_max_chars)
            chapter_summary = ""
            summary_status = "model_summary_empty"
            strict_mode = options.strict_mode
            chapter_max_tokens = options.chapter_max_tokens or self._summary_max_tokens()
            while True:
                try:
                    chapter_summary = await self._summarize_text(
                        chapter_content,
                        options,
                        max_tokens=chapter_max_tokens,
                        runtime_registry=runtime_registry,
                        chapter_state=chapter_state,
                        cancel_on_error=not strict_mode,
                        retry_limit_override=None if strict_mode else 0,
                    )
                    if chapter_summary:
                        summary_status = "ok"
                        break
                    if not strict_mode:
                        break
                    logger.warning(
                        "Chapter summary empty in strict mode, retrying: %s (index=%d)",
                        chapter.title,
                        index,
                    )
                except Exception as exc:
                    summary_status = "model_summary_error"
                    if not strict_mode:
                        logger.warning(
                            "Chapter summary failed in non-strict mode, skip chapter: %s (index=%d) - %s",
                            chapter.title,
                            index,
                            exc,
                        )
                        chapter_summary = ""
                        break
                    logger.warning(
                        "Chapter summary failed in strict mode, retrying: %s (index=%d) - %s",
                        chapter.title,
                        index,
                        exc,
                    )

                if strict_mode:
                    await asyncio.sleep(max(1.0, float(options.strict_retry_interval or 2.0)))

            if not chapter_summary:
                logger.warning(
                    "Chapter summary empty, skipping: %s (index=%d) reason=%s",
                    chapter.title,
                    index,
                    summary_status,
                )
                return "", [], True, summary_status

            extracted_characters: list[dict] = []
            if self.config.storage.characters_enabled:
                logger.info("Character extraction start: %s (index=%d)", chapter.title, index)
                char_result = await self._extract_characters_segments(chapter_content, options)
                if isinstance(char_result, dict) and "characters" in char_result:
                    extracted_characters.extend(char_result["characters"])
                elif isinstance(char_result, list):
                    extracted_characters.extend(char_result)
                logger.info("Character extraction done: %s (index=%d)", chapter.title, index)

            return chapter_summary, extracted_characters, False, summary_status

        while next_pending < len(pending_indices) or active:
            while next_pending < len(pending_indices) and len(active) < chapter_window:
                index = pending_indices[next_pending]
                chapter = chapters[index]
                future: asyncio.Task[Any] = asyncio.create_task(_run_one(index, chapter))
                active[future] = index
                runtime_registry.reduce_futures[future] = chapter.chapter_id
                runtime_registry.reduce_slots[chapter.chapter_id] = future
                next_pending += 1

            if not active:
                break

            done, _ = await asyncio.wait(set(active.keys()), return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                chapter = chapters[index]
                runtime_registry.reduce_futures.pop(future, None)
                runtime_registry.reduce_slots.pop(chapter.chapter_id, None)

                chapter_state = work_state_map.chapters.get(chapter.chapter_id)
                try:
                    summary_text, characters, failed, summary_status = future.result()
                except Exception as exc:
                    logger.error("Chapter task failed: %s (index=%d) - %s", chapter.title, index, exc)
                    summary_text, characters, failed, summary_status = "", [], True, "model_summary_error"

                summary_by_id[chapter.chapter_id] = self._build_chapter_summary_item(
                    chapter,
                    summary_text,
                    summary_status,
                )
                if summary_text.strip():
                    effective_summary_ids.add(chapter.chapter_id)
                else:
                    effective_summary_ids.discard(chapter.chapter_id)
                if characters:
                    all_characters.extend(characters)

                if chapter_state:
                    chapter_state.status = "FAILED" if failed else "DONE"
                    chapter_state.chapter_summary = summary_text
                    chapter_state.touch()

                ordered = self._ordered_chapter_summaries(chapters, summary_by_id)
                checkpoint = self._contiguous_done_index(chapters, summary_by_id)
                contiguous_summaries = self._contiguous_prefix_summaries(chapters, summary_by_id)
                self._write_partial_progress(
                    resolved_id,
                    resolved_title,
                    source_path,
                    chapters,
                    checkpoint,
                    contiguous_summaries,
                    all_characters,
                )

        return self._ordered_chapter_summaries(chapters, summary_by_id), all_characters

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
        analysis_progress = {
            "chapter_index": chapter_index,
            "chapter_summaries": chapter_summaries,
            "characters": characters,
        }
        progress_state = {
            "analysis_progress": analysis_progress
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
            chapter_max_tokens=self._summary_max_tokens(),
            strict_retry_interval=max(2.0, float(self.config.storage.segment_retry_interval or 1.0)),
        )

    async def _summarize_text(
        self,
        text: str,
        options: AnalysisOptions,
        max_tokens: Optional[int],
        runtime_registry: Optional[RuntimeTaskRegistry] = None,
        chapter_state: Optional[ChapterTaskState] = None,
        cancel_on_error: bool = False,
        retry_limit_override: Optional[int] = None,
    ) -> str:
        segments = self._segment_text(text, options)
        if not segments:
            return ""
        if chapter_state is not None:
            chapter_state.segment_texts = list(segments)
            chapter_state.status = "SEGMENTS_READY"
            chapter_state.touch()
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
            runtime_registry=runtime_registry,
            chapter_state=chapter_state,
            cancel_on_error=cancel_on_error,
            retry_limit_override=retry_limit_override,
        )
        cleaned = [
            text
            for item in summaries
            if item
            for text in [self._sanitize_summary_text(self._extract_summary(item))]
            if text
        ]
        combined = "\n".join(part for part in cleaned if part)
        return await self._summarize_until_fit(
            combined,
            options,
            max_tokens,
            cancel_on_error=cancel_on_error,
            retry_limit_override=retry_limit_override,
        )

    async def _summarize_items(
        self,
        items: list[str],
        options: AnalysisOptions,
        max_tokens: Optional[int],
        cancel_on_error: bool = False,
        retry_limit_override: Optional[int] = None,
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
            cancel_on_error=cancel_on_error,
            retry_limit_override=retry_limit_override,
        )
        cleaned = [
            text
            for item in summaries
            if item
            for text in [self._sanitize_summary_text(self._extract_summary(item))]
            if text
        ]
        combined = "\n".join(part for part in cleaned if part)
        return await self._summarize_until_fit(
            combined,
            options,
            max_tokens,
            cancel_on_error=cancel_on_error,
            retry_limit_override=retry_limit_override,
        )

    async def _summarize_until_fit(
        self,
        combined: str,
        options: AnalysisOptions,
        max_tokens: Optional[int],
        cancel_on_error: bool = False,
        retry_limit_override: Optional[int] = None,
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
                cancel_on_error=cancel_on_error,
                retry_limit_override=retry_limit_override,
            )
            cleaned = [
                text
                for item in summaries
                if item
                for text in [self._sanitize_summary_text(self._extract_summary(item))]
                if text
            ]
            new_combined = "\n".join(part for part in cleaned if part)
            if not new_combined or len(new_combined) >= len(combined):
                break
            combined = new_combined
            attempts += 1
        combined = truncate_text(combined, max_chars=max_chars)
        result = await self._analyze_plot_with_max_tokens(combined, max_tokens)
        return self._sanitize_summary_text(self._extract_summary(result))

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
        runtime_registry: Optional[RuntimeTaskRegistry] = None,
        chapter_state: Optional[ChapterTaskState] = None,
        cancel_on_error: bool = False,
        retry_limit_override: Optional[int] = None,
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
        effective_retry_limit = (
            options.retry_limit if retry_limit_override is None else max(0, int(retry_limit_override))
        )
        chapter_label = ""
        if chapter_state is not None:
            chapter_label = f"{chapter_state.title} (index={chapter_state.chapter_index})"
        logger.info(
            "Map with limits: total=%d concurrency=%d qps=%.2f retries=%d fail_fast=%s chapter=%s",
            total,
            concurrency,
            options.qps_limit,
            effective_retry_limit,
            cancel_on_error,
            chapter_label or "-",
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
                        "Segment skipped: %d/%d chapter=%s reason=empty_input",
                        index + 1,
                        total,
                        chapter_label or "-",
                    )
                    return {"summary": "", "summary_status": "empty_input"}
                for attempt in range(effective_retry_limit + 1):
                    start_time = time.monotonic()
                    try:
                        await limiter.wait()
                        # func 返回 AnalysisResult，我们需要 .content
                        result = await func(normalized_item)
                        elapsed = time.monotonic() - start_time
                        provider_type = getattr(result, "provider_type", None)
                        model_name = getattr(result, "model_name", None)
                        provider_desc = None
                        if provider_type and model_name:
                            provider_desc = f"{provider_type}:{model_name}"
                        elif provider_type:
                            provider_desc = str(provider_type)
                        logger.info(
                            "Segment done: %d/%d attempt=%d elapsed=%.2fs provider=%s chapter=%s",
                            index + 1,
                            total,
                            attempt + 1,
                            elapsed,
                            provider_desc or "unknown",
                            chapter_label or "-",
                        )
                        # 提取 content 字段
                        return result.content if hasattr(result, 'content') else result
                    except Exception as exc:
                        elapsed = time.monotonic() - start_time
                        provider_hint = None
                        if hasattr(exc, "args") and exc.args:
                            provider_hint = str(exc.args[0])
                        error_text = self._compact_error_text(provider_hint or exc)
                        logger.warning(
                            "Segment failed: %d/%d attempt=%d elapsed=%.2fs chapter=%s error=%s",
                            index + 1,
                            total,
                            attempt + 1,
                            elapsed,
                            chapter_label or "-",
                            error_text,
                        )
                        if attempt >= effective_retry_limit:
                            raise
                        await asyncio.sleep(options.retry_interval)

        results: list[Any] = [None] * total
        active: dict[asyncio.Task[Any], int] = {}
        next_to_schedule = 0

        while next_to_schedule < total or active:
            while next_to_schedule < total and len(active) < concurrency:
                item_index, item_value = indexed_items[next_to_schedule]
                future: asyncio.Task[Any] = asyncio.create_task(task(item_value, item_index))
                active[future] = item_index
                if runtime_registry is not None and chapter_state is not None:
                    slot = (chapter_state.chapter_id, item_index)
                    runtime_registry.segment_futures[future] = slot
                    runtime_registry.segment_slots[slot] = future
                    runtime_registry.last_tick_at = max(runtime_registry.last_tick_at, chapter_state.updated_at)
                next_to_schedule += 1

            if active:
                logger.info(
                    "Event loop tick: active=%d scheduled=%d/%d",
                    len(active),
                    next_to_schedule,
                    total,
                )

            if not active:
                break

            wait_mode = asyncio.FIRST_EXCEPTION if cancel_on_error else asyncio.FIRST_COMPLETED
            done, _ = await asyncio.wait(set(active.keys()), return_when=wait_mode)
            if cancel_on_error and any(f.done() and f.exception() is not None for f in done):
                for pending_future in list(active.keys()):
                    if pending_future not in done and not pending_future.done():
                        pending_future.cancel()
            for future in done:
                item_index = active.pop(future)
                if runtime_registry is not None and chapter_state is not None:
                    slot = runtime_registry.segment_futures.pop(future, None)
                    if slot is not None:
                        runtime_registry.segment_slots.pop(slot, None)
                try:
                    results[item_index] = future.result()
                except Exception as exc:
                    results[item_index] = exc
        
        # 检查是否有失败的任务
        errors = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
        if errors:
            error_details = []
            for idx, exc in errors:
                provider_hint = "unknown"
                if hasattr(exc, "args") and exc.args:
                    provider_hint = str(exc.args[0])
                error_details.append(f"  - Segment {idx+1}/{total}: {provider_hint}")
            prefix = f"Failed {len(errors)}/{total} segments"
            if chapter_label:
                prefix += f" for {chapter_label}"
            error_msg = prefix + ":\n" + "\n".join(error_details)
            logger.error(error_msg)
            # 抛出统一异常，避免第三方异常构造参数不匹配
            first_error = errors[0][1]
            raise RuntimeError(error_msg) from first_error
        
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
