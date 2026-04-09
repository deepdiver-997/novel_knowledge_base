from novel_kb.config.config_manager import ConfigManager
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.services.ingest_service import IngestService as SyncIngestService
from novel_kb.services.ingest_service_async import IngestService as AsyncIngestService


def test_segment_text_does_not_emit_empty_chunks_on_exact_boundary():
    config = ConfigManager.load_config()
    service = SyncIngestService(config, None, NovelRepository(config.storage.data_dir))

    text = "a" * config.storage.analysis_max_chars
    segments = service._segment_text(text, service._resolve_analysis_options(None))

    assert segments == [text]
    assert all(segment.strip() for segment in segments)


def test_summary_status_classification_distinguishes_empty_sources():
    for service in (SyncIngestService, AsyncIngestService):
        assert service._chapter_summary_status("", "") == "chapter_content_empty"
        assert service._chapter_summary_status("正文内容", "") == "model_summary_empty"
        assert service._chapter_summary_status("正文内容", "总结结果") == "ok"
        assert service._chapter_summary_status("正文内容", "", had_error=True) == "model_summary_error"
