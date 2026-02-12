from typing import Dict, List, Optional

from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.config.config_schema import StorageConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.utils.segment import split_paragraphs
from novel_kb.utils.text import truncate_text
from novel_kb.utils.vector import cosine_similarity


class SearchService:
    def __init__(
        self,
        repository: NovelRepository,
        storage_config: StorageConfig,
        embedding_builder: Optional[EmbeddingBuilder] = None,
    ) -> None:
        self.repository = repository
        self.storage_config = storage_config
        self.embedding_builder = embedding_builder

    def search_novels(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        if self._can_embed():
            return self._search_novels_semantic(query, k)
        return self._search_novels_keyword(query, k)

    def search_chapters(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> List[Dict[str, object]]:
        if self._can_embed() and self.storage_config.embed_chapters:
            return self._search_chapters_semantic(query, k, novel_id)
        return self._search_chapters_keyword(query, k, novel_id)

    def search_paragraphs(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> List[Dict[str, object]]:
        if not self.storage_config.paragraph_enabled:
            return []
        if self._can_embed() and self.storage_config.paragraph_semantic_enabled:
            return self._search_paragraphs_semantic(query, k, novel_id)
        return self._search_paragraphs_keyword(query, k, novel_id)

    def _search_novels_semantic(self, query: str, k: int) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_novels_keyword(query, k)

        results: List[Dict[str, object]] = []
        for record in self.repository.list_novels():
            summary = record.summary or ""
            if not summary:
                continue
            embedding = record.summary_embedding or self._embed_text(summary)
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            results.append({
                "novel_id": record.novel_id,
                "title": record.title,
                "score": score,
                "summary": truncate_text(summary, max_chars=400),
                "features": record.features,
                "characters": record.characters,
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_chapters_semantic(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_chapters_keyword(query, k, novel_id)

        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)
        for record in records:
            for chapter in record.chapters:
                if not chapter.content:
                    continue
                embedding = chapter.embedding
                if not embedding:
                    continue
                score = cosine_similarity(query_embedding, embedding)
                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "score": score,
                    "snippet": truncate_text(chapter.content, max_chars=300),
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_novels_keyword(self, query: str, k: int) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        for record in self.repository.list_novels():
            summary = record.summary or ""
            content = summary or (record.chapters[0].content if record.chapters else "")
            score = content.lower().count(query_lower)
            if score == 0:
                continue
            results.append({
                "novel_id": record.novel_id,
                "title": record.title,
                "score": score,
                "summary": truncate_text(content, max_chars=400),
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_chapters_keyword(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)
        for record in records:
            for chapter in record.chapters:
                content = chapter.content or ""
                score = content.lower().count(query_lower)
                if score == 0:
                    continue
                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "score": score,
                    "snippet": truncate_text(content, max_chars=300),
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_paragraphs_keyword(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)
        for record in records:
            for chapter in record.chapters:
                paragraphs = split_paragraphs(
                    chapter.content,
                    min_len=self.storage_config.paragraph_min_chars,
                )
                for index, paragraph in enumerate(paragraphs):
                    score = paragraph.lower().count(query_lower)
                    if score == 0:
                        continue
                    results.append({
                        "novel_id": record.novel_id,
                        "title": record.title,
                        "chapter_id": chapter.chapter_id,
                        "chapter_title": chapter.title,
                        "paragraph_index": index,
                        "score": score,
                        "snippet": truncate_text(paragraph, max_chars=300),
                    })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_paragraphs_semantic(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_paragraphs_keyword(query, k, novel_id)

        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)
        for record in records:
            for chapter in record.chapters:
                paragraphs = split_paragraphs(
                    chapter.content,
                    min_len=self.storage_config.paragraph_min_chars,
                )
                for index, paragraph in enumerate(paragraphs):
                    embedding = self._embed_text(paragraph)
                    if not embedding:
                        continue
                    score = cosine_similarity(query_embedding, embedding)
                    results.append({
                        "novel_id": record.novel_id,
                        "title": record.title,
                        "chapter_id": chapter.chapter_id,
                        "chapter_title": chapter.title,
                        "paragraph_index": index,
                        "score": score,
                        "snippet": truncate_text(paragraph, max_chars=300),
                    })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _filtered_records(self, novel_id: Optional[str]):
        if novel_id:
            return [self.repository.load_novel(novel_id)]
        return self.repository.list_novels()

    def _embed_text(self, text: str) -> List[float]:
        if not self.embedding_builder:
            return []
        payload = truncate_text(text, max_chars=self.storage_config.embedding_max_chars)
        embedding = self.embedding_builder.build(payload)
        return embedding.vector

    def _can_embed(self) -> bool:
        return self.storage_config.embedding_enabled and self.embedding_builder is not None
