import datetime
from pathlib import Path
from typing import Optional

from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.knowledge_base.schemas import ChapterRecord, NovelRecord
from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.llm.provider import LLMProvider
from novel_kb.parsers.factory import ParserFactory
from novel_kb.segmenters.chapter_segmenter import ChapterSegmenter
from novel_kb.utils.text import truncate_text


class IngestService:
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
    ) -> NovelRecord:
        parser = self.parser_factory.get_parser(file_path)
        document = parser.parse(file_path)

        resolved_title = title or document.title or Path(file_path).stem
        resolved_id = novel_id or self._slugify(resolved_title)

        if not overwrite and self.repository.exists(resolved_id):
            return self.repository.load_novel(resolved_id)

        chapters = self._segment_document(document)
        summary, features, characters = self._analyze_document(document.content)
        if not summary:
            summary = self._fallback_summary(chapters)

        summary_embedding, chapter_embeddings = self._build_embeddings(summary, chapters)

        record = NovelRecord(
            novel_id=resolved_id,
            title=resolved_title,
            summary=summary,
            features=features,
            characters=characters,
            chapters=chapters,
            summary_embedding=summary_embedding,
            metadata={"source_path": str(Path(file_path).expanduser())},
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
        )
        self.repository.save_novel(record)
        return record

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

    def _analyze_document(self, content: str) -> tuple[str, list[str], list[dict]]:
        if not self.provider or not self.config.storage.analysis_enabled:
            return "", [], []
        prompt_text = truncate_text(content, max_chars=self.config.storage.analysis_max_chars)
        try:
            plot = self.provider.analyze_plot(prompt_text).content
            characters = self.provider.analyze_characters(prompt_text).content
            features = []
            if self.config.storage.features_enabled:
                features = self.provider.extract_features(prompt_text).content
        except Exception:
            return "", [], []

        summary = plot.get("summary") if isinstance(plot, dict) else plot
        feature_list = features.get("features") if isinstance(features, dict) else features
        character_list = characters.get("characters") if isinstance(characters, dict) else characters

        return (
            summary or "",
            feature_list if isinstance(feature_list, list) else [],
            character_list if isinstance(character_list, list) else [],
        )

    def _build_embeddings(
        self,
        summary: str,
        chapters: list[ChapterRecord],
    ) -> tuple[Optional[list[float]], dict[str, list[float]]]:
        if not self.embedding_builder:
            return None, {}

        summary_embedding = None
        if self.config.storage.embed_summary and summary:
            embedding_text = truncate_text(summary, max_chars=self.config.storage.embedding_max_chars)
            summary_embedding = self.embedding_builder.build(embedding_text).vector

        chapter_embeddings: dict[str, list[float]] = {}
        if self.config.storage.embed_chapters:
            for chapter in chapters:
                embedding_text = truncate_text(chapter.content, max_chars=self.config.storage.embedding_max_chars)
                chapter_embeddings[chapter.chapter_id] = self.embedding_builder.build(embedding_text).vector

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
