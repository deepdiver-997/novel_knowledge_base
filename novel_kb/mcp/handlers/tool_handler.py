from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.llm.factory import LLMFactory
from novel_kb.llm.noop_provider import NoOpProvider
from novel_kb.llm.provider import LLMProvider
from novel_kb.services.export_service import ExportService
from novel_kb.services.ingest_service import IngestService
from novel_kb.services.search_service import SearchService
from novel_kb.utils.text import truncate_text


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable


class ToolHandler:
    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self.config = config
        self.repository = NovelRepository(config.storage.data_dir)
        self.provider = self._build_provider()
        self.ingest_service = IngestService(config, self.provider, self.repository)
        self.export_service = ExportService(self.repository, EmbeddingBuilder(self.provider))
        self.search_service = SearchService(self.repository, config.storage, EmbeddingBuilder(self.provider))
        self.tools: List[ToolSpec] = [
            ToolSpec("ingest_novel_file", "Ingest a local novel file", self.ingest_novel_file),
            ToolSpec("list_novels", "List indexed novels", self.list_novels),
            ToolSpec("export_content_chunks", "Export chunks for novel_mcp", self.export_content_chunks),
            ToolSpec("search_novel", "Search novel content", self.search_novel),
            ToolSpec("search_chapters", "Search within chapters", self.search_chapters),
            ToolSpec("search_paragraphs", "Search within paragraphs", self.search_paragraphs),
            ToolSpec("recommend_novels", "Recommend novels by preference", self.recommend_novels),
            ToolSpec("analyze_novel", "Analyze novel features", self.analyze_novel),
            ToolSpec("extract_characters", "Extract character relationships", self.extract_characters),
            ToolSpec("get_summary", "Summarize novel plot", self.get_summary),
            ToolSpec("health_check", "Check knowledge base status", self.health_check),
        ]

    def register(self, server) -> None:
        if hasattr(server, "tool"):
            for tool in self.tools:
                server.tool(name=tool.name, description=tool.description)(tool.func)
            return
        if hasattr(server, "add_tool"):
            for tool in self.tools:
                server.add_tool(tool.name, tool.func, tool.description)
            return
        raise RuntimeError("Unsupported MCP server interface")

    def ingest_novel_file(self, file_path: str, novel_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, object]:
        record = self.ingest_service.ingest_file(file_path, novel_id=novel_id, title=title)
        return {
            "novel_id": record.novel_id,
            "title": record.title,
            "chapters": len(record.chapters),
        }

    def list_novels(self) -> Dict[str, object]:
        return {"novels": self.export_service.export_novel_list()}

    def export_content_chunks(
        self,
        novel_id: str,
        include_chapters: bool = True,
        include_embeddings: bool = False,
    ) -> Dict[str, object]:
        chunks = self.export_service.export_content_chunks(
            novel_id,
            include_chapters=include_chapters,
            include_embeddings=include_embeddings,
        )
        return {"chunks": chunks}

    def search_novel(self, query: str, k: int = 5) -> Dict[str, object]:
        return {"results": self.search_service.search_novels(query, k=k)}

    def search_chapters(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> Dict[str, object]:
        results = self.search_service.search_chapters(query, k=k, novel_id=novel_id)
        return {"results": results}

    def search_paragraphs(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> Dict[str, object]:
        results = self.search_service.search_paragraphs(query, k=k, novel_id=novel_id)
        return {"results": results}

    def recommend_novels(self, preference_text: str, k: int = 5) -> Dict[str, object]:
        return {"results": self.search_service.search_novels(preference_text, k=k)}

    def analyze_novel(self, text: str) -> Dict[str, object]:
        if (
            not self.provider
            or not self.config.storage.analysis_enabled
            or not self.config.storage.features_enabled
        ):
            return {"features": []}
        result = self.provider.extract_features(
            truncate_text(text, max_chars=self.config.storage.analysis_max_chars)
        )
        return {"features": result.content}

    def extract_characters(self, text: str) -> Dict[str, object]:
        if not self.provider or not self.config.storage.analysis_enabled:
            return {"characters": []}
        result = self.provider.analyze_characters(
            truncate_text(text, max_chars=self.config.storage.analysis_max_chars)
        )
        return {"characters": result.content}

    def get_summary(self, novel_id: Optional[str] = None, text: Optional[str] = None) -> Dict[str, object]:
        if novel_id:
            record = self.repository.load_novel(novel_id)
            return {"summary": record.summary or truncate_text(record.chapters[0].content, max_chars=800)}
        if text and self.provider and self.config.storage.analysis_enabled:
            result = self.provider.analyze_plot(
                truncate_text(text, max_chars=self.config.storage.analysis_max_chars)
            )
            return {"summary": result.content}
        return {"summary": ""}

    def health_check(self) -> Dict[str, object]:
        provider_ok = False
        try:
            provider_ok = bool(self.provider.health_check())
        except Exception:
            provider_ok = False
        return {
            "provider": self.config.llm.provider,
            "provider_ok": provider_ok,
            "data_dir": str(self.config.storage.data_dir),
            "analysis_enabled": self.config.storage.analysis_enabled,
            "features_enabled": self.config.storage.features_enabled,
            "embedding_enabled": self.config.storage.embedding_enabled,
            "paragraph_enabled": self.config.storage.paragraph_enabled,
        }

    def _build_provider(self) -> LLMProvider:
        try:
            provider = LLMFactory.create(self.config.llm)
            if provider.health_check():
                return provider
        except Exception:
            pass
        return NoOpProvider()

