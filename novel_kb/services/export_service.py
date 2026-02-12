import datetime
from typing import Dict, List, Optional

from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.utils.text import truncate_text


class ExportService:
    def __init__(
        self,
        repository: NovelRepository,
        embedding_builder: Optional[EmbeddingBuilder] = None,
    ) -> None:
        self.repository = repository
        self.embedding_builder = embedding_builder

    def export_content_chunks(
        self,
        novel_id: str,
        include_chapters: bool = True,
        include_embeddings: bool = False,
    ) -> List[Dict[str, object]]:
        record = self.repository.load_novel(novel_id)
        chunks: List[Dict[str, object]] = []

        summary_content = record.summary or self._fallback_summary(record)
        chunks.append(
            self._build_chunk(
                chunk_id=f"{record.novel_id}:summary",
                content=summary_content,
                content_type="novel_summary",
                source_id=record.novel_id,
                source_title=record.title,
                metadata={"features": record.features, "characters": record.characters},
                created_at=record.created_at,
                include_embeddings=include_embeddings,
                existing_embedding=record.summary_embedding,
            )
        )

        if include_chapters:
            for index, chapter in enumerate(record.chapters):
                chunks.append(
                    self._build_chunk(
                        chunk_id=f"{record.novel_id}:{chapter.chapter_id}",
                        content=chapter.content,
                        content_type="chapter",
                        source_id=record.novel_id,
                        source_title=record.title,
                        metadata={
                            "chapter_id": chapter.chapter_id,
                            "chapter_title": chapter.title,
                            "chapter_index": index,
                            **(chapter.metadata or {}),
                        },
                        created_at=record.created_at,
                        include_embeddings=include_embeddings,
                        existing_embedding=chapter.embedding,
                    )
                )

        return chunks

    def export_novel_list(self) -> List[Dict[str, str]]:
        return [
            {"novel_id": record.novel_id, "title": record.title}
            for record in self.repository.list_novels()
        ]

    def _build_chunk(
        self,
        chunk_id: str,
        content: str,
        content_type: str,
        source_id: str,
        source_title: str,
        metadata: Dict[str, object],
        created_at: str,
        include_embeddings: bool,
        existing_embedding: Optional[List[float]] = None,
    ) -> Dict[str, object]:
        chunk = {
            "id": chunk_id,
            "content": content,
            "content_type": content_type,
            "source_id": source_id,
            "source_title": source_title,
            "metadata": metadata,
            "embedding": existing_embedding,
            "created_at": created_at or datetime.datetime.utcnow().isoformat() + "Z",
        }
        if include_embeddings and self.embedding_builder and chunk["embedding"] is None:
            embedding_text = truncate_text(content, max_chars=4000)
            embedding = self.embedding_builder.build(embedding_text)
            chunk["embedding"] = embedding.vector
        return chunk

    @staticmethod
    def _fallback_summary(record) -> str:
        if not record.chapters:
            return ""
        return truncate_text(record.chapters[0].content, max_chars=800)
