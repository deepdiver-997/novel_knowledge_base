import argparse
import asyncio
from pathlib import Path

from novel_kb.config.config_manager import ConfigManager
from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.llm.noop_provider import NoOpProvider
from novel_kb.llm.provider import LLMProvider
from novel_kb.mcp.server import NovelKBMCPServer
from novel_kb.services.epub_cleaner import EPUBCleaner
from novel_kb.services.ingest_service import AnalysisOptions, IngestService
from novel_kb.services.summary_audit import ActionMode, SuspiciousSummary, audit_summary_file
from novel_kb.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Novel knowledge base MCP server"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Config file path (default: ~/.novel_knowledge_base/config.yaml)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create default config file and exit",
    )

    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a novel file")
    ingest_parser.add_argument("file", type=str, help="Path to novel file")
    ingest_parser.add_argument("--novel-id", type=str, help="Override novel id")
    ingest_parser.add_argument("--title", type=str, help="Override novel title")
    ingest_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing record")
    _add_clean_args(ingest_parser)
    _add_segment_args(ingest_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume a partial ingest")
    resume_parser.add_argument("file", type=str, help="Path to novel file")
    resume_parser.add_argument("--novel-id", type=str, help="Override novel id")
    resume_parser.add_argument("--title", type=str, help="Override novel title")
    _add_clean_args(resume_parser)
    _add_segment_args(resume_parser)

    subparsers.add_parser("list", help="List ingested novels")

    # User management commands
    register_user_parser = subparsers.add_parser("register_user", help="Register a new user")
    register_user_parser.add_argument("name", type=str, help="Name for the new user")
    register_user_parser.add_argument("api_key", type=str, help="API key for the user")

    delete_user_parser = subparsers.add_parser("delete_user", help="Delete a user and their KB ownership records")
    delete_user_parser.add_argument("name", type=str, help="Name of the user to delete")

    assign_kb_parser = subparsers.add_parser("assign_kb", help="Assign a KB file to a user")
    assign_kb_parser.add_argument("name", type=str, help="Name of the user")
    assign_kb_parser.add_argument("novel_id", type=str, help="Novel ID to assign")

    list_user_kb_parser = subparsers.add_parser("list_user_kb", help="List KB files accessible by a user")
    list_user_kb_parser.add_argument("name", type=str, help="Name of the user")

    reindex_parser = subparsers.add_parser("reindex", help="Re-generate vector embeddings for existing novels")
    reindex_parser.add_argument(
        "--novel-id",
        type=str,
        default=None,
        help="Specific novel ID to reindex (default: all novels)",
    )
    reindex_parser.add_argument(
        "--novel-ids",
        type=str,
        nargs="+",
        default=None,
        help="Multiple novel IDs to reindex",
    )
    reindex_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex even if chunks already exist",
    )

    audit_parser = subparsers.add_parser(
        "audit-summaries",
        help="Audit chapter summaries with fingerprint library",
    )
    audit_parser.add_argument("file", type=str, help="Path to <novel>.json or <novel>.progress.json")
    audit_parser.add_argument(
        "--fingerprints",
        type=str,
        default="wrong_fliter.txt",
        help="Fingerprint text file path (one phrase per line)",
    )
    audit_parser.add_argument(
        "--action",
        choices=["report", "delete", "confirm-delete"],
        default="report",
        help="report: only list suspicious; delete: remove all suspicious; confirm-delete: ask one by one",
    )
    audit_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.72,
        help="Similarity threshold against fingerprint lines (0-1)",
    )
    audit_parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Flag summaries shorter than this normalized length",
    )
    audit_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Only keep suspicious entries with score >= this value",
    )
    audit_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable .bak backup when action deletes entries",
    )

    args = parser.parse_args()

    if args.init:
        path = ConfigManager.ensure_default(args.config)
        print(f"Config initialized: {path}")
        return

    config = ConfigManager.load_config(args.config)

    if args.command == "ingest":
        should_clean = bool(getattr(args, "clean", False) or getattr(args, "clean_only", False))
        target_file = _prepare_input_file(args.file, should_clean)
        if bool(getattr(args, "clean_only", False)):
            print(f"Cleaned EPUB: {target_file}")
            return
        repository = NovelRepository(config.storage.data_dir)
        provider = _build_provider(config)
        service = IngestService(config, provider, repository)
        analysis_options = _build_analysis_options(args, config)
        record = service.ingest_file(
            target_file,
            novel_id=args.novel_id,
            title=args.title,
            overwrite=bool(args.overwrite),
            analysis_options=analysis_options,
        )
        print(f"Ingested: {record.title} ({record.novel_id})")
        print(f"Chapters: {len(record.chapters)}")
        return

    if args.command == "resume":
        should_clean = bool(getattr(args, "clean", False) or getattr(args, "clean_only", False))
        target_file = _prepare_input_file(args.file, should_clean)
        if bool(getattr(args, "clean_only", False)):
            print(f"Cleaned EPUB: {target_file}")
            return
        repository = NovelRepository(config.storage.data_dir)
        provider = _build_provider(config)
        service = IngestService(config, provider, repository)
        analysis_options = _build_analysis_options(args, config)
        record = service.resume_file(
            target_file,
            novel_id=args.novel_id,
            title=args.title,
            analysis_options=analysis_options,
        )
        print(f"Resumed: {record.title} ({record.novel_id})")
        print(f"Chapters: {len(record.chapters)}")
        return

    if args.command == "list":
        repository = NovelRepository(config.storage.data_dir)
        novels = repository.list_novels()
        if not novels:
            print("No novels found")
            return
        for item in novels:
            print(f"{item.novel_id}\t{item.title}")
        return

    # User management commands
    if args.command == "register_user":
        from novel_kb.auth.user_manager import UserManager
        user_mgr = UserManager()
        try:
            user_id = user_mgr.register_user(args.name, args.api_key)
            print(f"User registered: name={args.name}, user_id={user_id}")
        except ValueError as e:
            print(f"Error: {e}")
        return

    if args.command == "delete_user":
        from novel_kb.auth.user_manager import UserManager
        user_mgr = UserManager()
        if user_mgr.delete_user(args.name):
            print(f"User deleted: name={args.name}")
        else:
            print(f"Error: User not found: {args.name}")
        return

    if args.command == "assign_kb":
        from novel_kb.auth.user_manager import UserManager
        user_mgr = UserManager()
        try:
            if user_mgr.assign_kb(args.name, args.novel_id):
                print(f"KB assigned: name={args.name}, novel_id={args.novel_id}")
            else:
                print(f"KB already assigned: name={args.name}, novel_id={args.novel_id}")
        except ValueError as e:
            print(f"Error: {e}")
        return

    if args.command == "list_user_kb":
        from novel_kb.auth.user_manager import UserManager
        user_mgr = UserManager()
        kb_list = user_mgr.list_user_kb(args.name)
        if kb_list:
            for novel_id in kb_list:
                print(novel_id)
        else:
            print(f"No KB files found for user: {args.name}")
        return

    if args.command == "audit-summaries":
        target_path = Path(args.file).expanduser()
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {target_path}")
        fingerprint_path = Path(args.fingerprints).expanduser()

        def _confirm(item: SuspiciousSummary) -> bool:
            print("\n---")
            print(f"index={item.index} chapter_id={item.chapter_id} title={item.title}")
            print(f"score={item.score} reasons={'; '.join(item.reasons)}")
            print(f"summary={item.summary}")
            choice = input("Delete this summary entry? [y/N]: ").strip().lower()
            return choice in {"y", "yes"}

        action: ActionMode = args.action
        suspicious, removed_count = audit_summary_file(
            file_path=target_path,
            fingerprint_path=fingerprint_path,
            action=action,
            min_length=max(1, int(args.min_length)),
            similarity_threshold=max(0.0, min(1.0, float(args.similarity_threshold))),
            min_score=max(0.0, float(args.min_score)),
            backup=not bool(args.no_backup),
            confirm_callback=_confirm if action == "confirm-delete" else None,
        )

        print(f"Suspicious summaries: {len(suspicious)}")
        for item in suspicious[:30]:
            reasons = "; ".join(item.reasons)
            preview = " ".join(item.summary.split())[:120]
            print(f"- idx={item.index} id={item.chapter_id} score={item.score} reasons={reasons}")
            print(f"  title={item.title}")
            print(f"  summary={preview}")
        if len(suspicious) > 30:
            print(f"... omitted {len(suspicious) - 30} entries")

        if action in {"delete", "confirm-delete"}:
            print(f"Removed entries: {removed_count}")
            if not args.no_backup and removed_count > 0:
                print(f"Backup file: {target_path.with_suffix(target_path.suffix + '.bak')}")
        return

    # reindex 命令 - 重新生成向量嵌入
    if args.command == "reindex":
        from novel_kb.services.embedding_service import EmbeddingConfig, EmbeddingService
        from novel_kb.knowledge_base.vector_store import create_vector_store

        repository = NovelRepository(config.storage.data_dir)
        provider = _build_provider(config)
        vector_store = create_vector_store(
            store_type=config.storage.vector_db_type,
            persist_directory=config.storage.data_dir / "vector_store",
        )
        vector_store.initialize()

        embed_config = EmbeddingConfig(
            enabled=True,
            embed_summary=config.storage.embed_summary,
            embed_chapters=config.storage.embed_chapters,
            embed_plot_summaries=config.storage.embed_plot_summaries,
            embed_paragraphs=config.storage.embed_paragraphs,
            paragraph_min_chars=config.storage.paragraph_min_chars,
            paragraph_max_chars=config.storage.paragraph_max_chars,
            max_chars_per_embedding=config.storage.embedding_max_chars,
        )

        service = EmbeddingService(provider=provider, vector_store=vector_store, config=embed_config)

        # 确定要索引的小说
        novel_ids: list[str] | None = None
        if getattr(args, "novel_id", None):
            novel_ids = [args.novel_id]
        elif getattr(args, "novel_ids", None):
            novel_ids = args.novel_ids

        force = bool(getattr(args, "force", False))
        results = service.reindex_all_novels_sync(repository, force=force, novel_ids=novel_ids)

        total_success = sum(r[0] for r in results.values())
        total_skipped = sum(r[1] for r in results.values())
        print(f"Reindex complete: {len(results)} novels, {total_success} chunks indexed, {total_skipped} skipped")
        return

    server = NovelKBMCPServer(config)
    server.run()


def _build_provider(config: KnowledgeBaseConfig) -> LLMProvider:
    """构建 LLM Provider（优先使用 gateway 模式）"""
    if getattr(config.llm, "use_gateway", False):
        from novel_kb.gateway_client import GatewayClient
        gateway_url = getattr(config.llm, "gateway_url", "http://127.0.0.1:8747")
        gateway_tier = getattr(config.llm, "gateway_tier", "medium")
        # Gateway 支持 max_tokens 和 temperature 参数覆盖
        gateway_max_tokens = getattr(config.llm, "gateway_max_tokens", None)
        gateway_temperature = getattr(config.llm, "gateway_temperature", None)
        # 按文本长度自动选择 tier
        gateway_tier_short = getattr(config.llm, "gateway_tier_short", None)
        gateway_tier_long = getattr(config.llm, "gateway_tier_long", None)
        gateway_short_chars = getattr(config.llm, "gateway_short_chars", 1000)
        gateway_long_chars = getattr(config.llm, "gateway_long_chars", 4000)
        logger.info("Using Gateway client: url=%s tier=%s max_tokens=%s temperature=%s "
                     "tier_short=%s tier_long=%s",
                     gateway_url, gateway_tier, gateway_max_tokens, gateway_temperature,
                     gateway_tier_short, gateway_tier_long)
        return GatewayClient(
            base_url=gateway_url,
            tier=gateway_tier,
            max_tokens=gateway_max_tokens,
            temperature=gateway_temperature,
            tier_short=gateway_tier_short,
            tier_long=gateway_tier_long,
            short_text_chars=gateway_short_chars,
            long_text_chars=gateway_long_chars,
        )

    try:
        from novel_kb.llm.factory import LLMFactory
        from novel_kb.llm.provider_pool import ProviderPool

        providers_list = config.llm.get_providers()
        logger.info("Building providers: %s", providers_list)

        providers: list[LLMProvider] = []
        for provider_type in providers_list:
            models = config.llm.get_models_for_provider(provider_type)
            if not models:
                models = [None]
            for model_name in models:
                try:
                    provider = LLMFactory.create(
                        config.llm,
                        model_selection="fast",
                        provider_type=provider_type,
                        model_override=model_name,
                    )
                    if asyncio.run(provider.health_check()):
                        providers.append(provider)
                        if model_name:
                            logger.info("Provider %s model %s is healthy", provider_type, model_name)
                        else:
                            logger.info("Provider %s is healthy", provider_type)
                    else:
                        if model_name:
                            logger.warning("Provider %s model %s failed health check", provider_type, model_name)
                        else:
                            logger.warning("Provider %s failed health check", provider_type)
                except Exception as exc:
                    if model_name:
                        logger.error("Failed to create provider %s model %s: %s", provider_type, model_name, exc)
                    else:
                        logger.error("Failed to create provider %s: %s", provider_type, exc)

        if not providers:
            logger.error("No healthy providers available")
            return NoOpProvider()

        if len(providers) == 1:
            return providers[0]

        return ProviderPool(providers)

    except Exception as exc:
        logger.error("Provider creation failed: %s", exc)
        return NoOpProvider()



def _add_clean_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean EPUB first and ingest/resume with *_cleaned.epub",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean EPUB and output *_cleaned.epub, do not ingest/resume",
    )


def _prepare_input_file(file_path: str, clean: bool) -> str:
    if not clean:
        return file_path
    cleaned = EPUBCleaner.clean(file_path)
    logger.info("Using cleaned file for processing: %s", cleaned)
    return cleaned


def _add_segment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segment-min-chars", type=int, default=None, help="Min chars per segment")
    parser.add_argument("--segment-max-chars", type=int, default=None, help="Max chars per segment (0 = auto)")
    parser.add_argument("--segment-concurrency", type=int, default=None, help="Segment analysis concurrency")
    parser.add_argument("--segment-qps", type=float, default=None, help="Segment analysis QPS limit")
    parser.add_argument("--segment-retries", type=int, default=None, help="Segment analysis retry count")
    parser.add_argument("--segment-retry-interval", type=float, default=None, help="Segment analysis retry interval seconds")
    parser.add_argument(
        "--chapter-max-tokens",
        type=int,
        default=None,
        help="Max tokens for chapter summary calls (gateway max_tokens override)",
    )
    parser.add_argument(
        "--strict-retry-interval",
        type=float,
        default=None,
        help="Retry interval seconds used by strict-mode chapter summary loop",
    )
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        help="Strict finalize mode: all chapter summaries must be valid, then generate plot and overall summaries",
    )
    parser.add_argument(
        "--fingerprints",
        type=str,
        default=None,
        help="Fingerprint library file for strict-mode irrelevant-summary filtering",
    )
    parser.add_argument(
        "--audit-min-score",
        type=float,
        default=0.9,
        help="Strict-mode suspicious score threshold for regeneration",
    )
    parser.add_argument(
        "--audit-similarity-threshold",
        type=float,
        default=0.72,
        help="Strict-mode fingerprint similarity threshold",
    )
    parser.add_argument(
        "--audit-min-length",
        type=int,
        default=20,
        help="Strict-mode minimum normalized summary length",
    )


def _build_analysis_options(args: argparse.Namespace, config: "KnowledgeBaseConfig") -> AnalysisOptions:
    segment_min_chars = args.segment_min_chars if args.segment_min_chars is not None else config.storage.segment_min_chars
    segment_max_chars = args.segment_max_chars if args.segment_max_chars is not None else config.storage.segment_max_chars
    segment_concurrency = (
        args.segment_concurrency if args.segment_concurrency is not None else config.storage.segment_concurrency
    )
    segment_qps = args.segment_qps if args.segment_qps is not None else config.storage.segment_qps
    segment_retries = (
        args.segment_retries if args.segment_retries is not None else config.storage.segment_retries
    )
    segment_retry_interval = (
        args.segment_retry_interval
        if args.segment_retry_interval is not None
        else config.storage.segment_retry_interval
    )
    chapter_max_tokens = (
        args.chapter_max_tokens
        if getattr(args, "chapter_max_tokens", None) is not None
        else getattr(config.llm, "summary_max_tokens", None)
    )
    strict_retry_interval = (
        args.strict_retry_interval
        if getattr(args, "strict_retry_interval", None) is not None
        else max(2.0, float(config.storage.segment_retry_interval or 1.0))
    )
    return AnalysisOptions(
        segment_enabled=True,
        segment_min_chars=segment_min_chars,
        segment_max_chars=segment_max_chars or None,
        concurrency_limit=segment_concurrency,
        qps_limit=segment_qps,
        retry_limit=segment_retries,
        retry_interval=segment_retry_interval,
        strict_mode=bool(getattr(args, "strict_mode", False)),
        chapter_max_tokens=chapter_max_tokens,
        strict_retry_interval=max(1.0, float(strict_retry_interval)),
        fingerprint_path=getattr(args, "fingerprints", None),
        audit_min_score=max(0.0, float(getattr(args, "audit_min_score", 0.9))),
        audit_similarity_threshold=max(0.0, min(1.0, float(getattr(args, "audit_similarity_threshold", 0.72)))),
        audit_min_length=max(1, int(getattr(args, "audit_min_length", 20))),
    )


if __name__ == "__main__":
    main()
