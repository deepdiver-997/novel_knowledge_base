"""向量存储层

提供统一的向量存储接口，支持内存存储和 Chroma 持久化存储。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import chromadb

    from novel_kb.knowledge_base.schemas import ChapterRecord, NovelRecord


class ContentType(str, Enum):
    """内容的语义类型，用于过滤和路由"""
    CHAPTER_SUMMARY = "chapter_summary"
    PLOT_SUMMARY = "plot_summary"
    OVERALL_SUMMARY = "overall_summary"
    PARAGRAPH = "paragraph"


@dataclass
class Chunk:
    """可嵌入的文本块"""
    chunk_id: str
    novel_id: str
    novel_title: str
    content: str
    content_type: ContentType | str
    chapter_id: Optional[str] = None
    chapter_index: Optional[int] = None
    chapter_title: Optional[str] = None
    volume_title: Optional[str] = None
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    content: str
    content_type: str
    score: float
    novel_id: str
    novel_title: str
    chapter_id: Optional[str] = None
    chapter_index: Optional[int] = None
    chapter_title: Optional[str] = None
    volume_title: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def initialize(self) -> None:
        """初始化存储"""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """插入或更新 chunks"""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        content_type: Optional[ContentType | str] = None,
        novel_id: Optional[str] = None,
        volume_title: Optional[str] = None,
    ) -> list[SearchResult]:
        """向量相似度搜索"""
        raise NotImplementedError

    @abstractmethod
    def delete_novel(self, novel_id: str) -> None:
        """删除某本小说的所有 chunks"""
        raise NotImplementedError

    @abstractmethod
    def count(self, novel_id: Optional[str] = None) -> int:
        """统计 chunks 数量"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-Memory Implementation (simple, for testing)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(VectorStore):
    """内存向量存储，简单实现，用于测试或小数据量"""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def initialize(self) -> None:
        pass

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.embedding:
                self._chunks[chunk.chunk_id] = chunk

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        content_type: Optional[ContentType | str] = None,
        novel_id: Optional[str] = None,
        volume_title: Optional[str] = None,
    ) -> list[SearchResult]:
        scored: list[tuple[Chunk, float]] = []
        ct_value = content_type.value if isinstance(content_type, ContentType) else content_type

        for chunk in self._chunks.values():
            if ct_value and chunk.content_type != ct_value:
                continue
            if novel_id and chunk.novel_id != novel_id:
                continue
            if volume_title and chunk.volume_title != volume_title:
                continue
            if not chunk.embedding:
                continue

            score = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                content_type=c.content_type.value if isinstance(c.content_type, ContentType) else c.content_type,
                score=s,
                novel_id=c.novel_id,
                novel_title=c.novel_title,
                chapter_id=c.chapter_id,
                chapter_index=c.chapter_index,
                chapter_title=c.chapter_title,
                volume_title=c.volume_title,
                metadata=c.metadata,
            )
            for c, s in scored[:k]
        ]

    def delete_novel(self, novel_id: str) -> None:
        self._chunks = {k: v for k, v in self._chunks.items() if v.novel_id != novel_id}

    def count(self, novel_id: Optional[str] = None) -> int:
        if novel_id:
            return sum(1 for c in self._chunks.values() if c.novel_id == novel_id)
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Chroma Implementation
# ---------------------------------------------------------------------------

class ChromaVectorStore(VectorStore):
    """Chroma 持久化向量存储"""

    COLLECTION_NAME = "novel_chunks"

    def __init__(
        self,
        persist_directory: Path,
        collection_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_kwargs = collection_kwargs or {}
        self._client: Optional["chromadb.PersistentClient"] = None
        self._collection: Optional["chromadb.Collection"] = None

    def initialize(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata=self.collection_kwargs or None,
        )

    def _ensure_initialized(self) -> None:
        if self._collection is None:
            self.initialize()

    def _to_chroma_metadata(self, chunk: Chunk) -> dict[str, Any]:
        ct = chunk.content_type.value if isinstance(chunk.content_type, ContentType) else chunk.content_type
        metadata: dict[str, Any] = {
            "novel_id": chunk.novel_id,
            "novel_title": chunk.novel_title,
            "content_type": ct,
        }
        if chunk.chapter_id:
            metadata["chapter_id"] = chunk.chapter_id
        if chunk.chapter_index is not None:
            metadata["chapter_index"] = chunk.chapter_index
        if chunk.chapter_title:
            metadata["chapter_title"] = chunk.chapter_title
        if chunk.volume_title:
            metadata["volume_title"] = chunk.volume_title
        for k, v in chunk.metadata.items():
            if k not in metadata:
                metadata[k] = v
        return metadata

    def upsert(self, chunks: list[Chunk]) -> None:
        self._ensure_initialized()
        valid = [c for c in chunks if c.embedding]
        if not valid:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in valid],
            embeddings=[c.embedding for c in valid],
            documents=[c.content for c in valid],
            metadatas=[self._to_chroma_metadata(c) for c in valid],
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        content_type: Optional[ContentType | str] = None,
        novel_id: Optional[str] = None,
        volume_title: Optional[str] = None,
    ) -> list[SearchResult]:
        self._ensure_initialized()

        # Build where clause - ChromaDB requires $and for multiple conditions
        conditions = []
        if novel_id:
            conditions.append({"novel_id": {"$eq": novel_id}})
        if content_type:
            ct = content_type.value if isinstance(content_type, ContentType) else content_type
            conditions.append({"content_type": {"$eq": ct}})
        if volume_title:
            conditions.append({"volume_title": {"$eq": volume_title}})

        if len(conditions) == 0:
            where_clause = None
        elif len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
        )
        return self._parse_results(results, k)

    def _parse_results(self, results: dict, k: int) -> list[SearchResult]:
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = (results.get("distances", [[]])[0] or []) if results.get("distances") else [0.0] * len(ids)
        documents = (results.get("documents", [[]])[0] or []) if results.get("documents") else []
        metadatas = (results.get("metadatas", [[]])[0] or []) if results.get("metadatas") else []

        search_results: list[SearchResult] = []
        for i, chunk_id in enumerate(ids):
            if i >= k:
                break
            metadata = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 - distance

            search_results.append(SearchResult(
                chunk_id=chunk_id,
                content=doc,
                content_type=metadata.get("content_type", ""),
                score=score,
                novel_id=metadata.get("novel_id", ""),
                novel_title=metadata.get("novel_title", ""),
                chapter_id=metadata.get("chapter_id"),
                chapter_index=metadata.get("chapter_index"),
                chapter_title=metadata.get("chapter_title"),
                volume_title=metadata.get("volume_title"),
                metadata=metadata,
            ))
        return search_results

    def delete_novel(self, novel_id: str) -> None:
        self._ensure_initialized()
        self._collection.delete(where={"novel_id": novel_id})

    def count(self, novel_id: Optional[str] = None) -> int:
        self._ensure_initialized()
        if novel_id:
            results = self._collection.get(where={"novel_id": novel_id})
            return len(results["ids"]) if results else 0
        return self._collection.count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vector_store(
    store_type: str = "chroma",
    persist_directory: Optional[Path] = None,
    **kwargs,
) -> VectorStore:
    """工厂函数创建向量存储"""
    if store_type == "chroma":
        if not persist_directory:
            raise ValueError("Chroma requires persist_directory")
        return ChromaVectorStore(persist_directory=persist_directory, **kwargs)
    elif store_type == "memory":
        return InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
