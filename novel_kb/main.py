import argparse

from novel_kb.config.config_manager import ConfigManager
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.llm.factory import LLMFactory
from novel_kb.llm.noop_provider import NoOpProvider
from novel_kb.mcp.server import NovelKBMCPServer
from novel_kb.services.ingest_service import IngestService


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

    subparsers.add_parser("list", help="List ingested novels")

    args = parser.parse_args()

    if args.init:
        path = ConfigManager.ensure_default(args.config)
        print(f"Config initialized: {path}")
        return

    config = ConfigManager.load_config(args.config)

    if args.command == "ingest":
        repository = NovelRepository(config.storage.data_dir)
        provider = _build_provider(config)
        service = IngestService(config, provider, repository)
        record = service.ingest_file(
            args.file,
            novel_id=args.novel_id,
            title=args.title,
            overwrite=bool(args.overwrite),
        )
        print(f"Ingested: {record.title} ({record.novel_id})")
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

    server = NovelKBMCPServer(config)
    server.run()


def _build_provider(config) -> object:
    try:
        provider = LLMFactory.create(config.llm)
        if provider.health_check():
            return provider
    except Exception:
        return NoOpProvider()
    return NoOpProvider()


if __name__ == "__main__":
    main()
