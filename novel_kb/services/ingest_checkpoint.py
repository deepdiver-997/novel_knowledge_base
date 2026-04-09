"""摄入进度状态管理"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

from novel_kb.knowledge_base.schemas import ChapterRecord


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
            self.updated_at = time.monotonic()

    def touch(self) -> None:
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
            order = [
                chapter.chapter_id
                for chapter in sorted(chapter_states.values(), key=lambda item: item.chapter_index)
            ]

        return WorkStateMap(
            chapters=chapter_states,
            inflight_segment_tasks=int(data.get("inflight_segment_tasks", 0) or 0),
            inflight_reduce_tasks=int(data.get("inflight_reduce_tasks", 0) or 0),
            order=order,
        )

    @staticmethod
    def create(chapters: list[ChapterRecord]) -> "WorkStateMap":
        """Create a new WorkStateMap from chapters."""
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
        now = time.monotonic()
        return RuntimeTaskRegistry(
            segment_futures={},
            reduce_futures={},
            segment_slots={},
            reduce_slots={},
            created_at=now,
            last_tick_at=now,
        )
