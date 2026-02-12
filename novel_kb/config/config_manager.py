import os
from pathlib import Path
from typing import Optional

import yaml

from novel_kb.config.config_schema import KnowledgeBaseConfig, LLMConfig, ParserConfig, StorageConfig


class ConfigManager:
    DEFAULT_CONFIG_DIR = Path.home() / ".novel_knowledge_base"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

    @staticmethod
    def ensure_default(config_path: Optional[str] = None) -> Path:
        path = Path(config_path) if config_path else ConfigManager.DEFAULT_CONFIG_FILE
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "llm": {
                "provider": "ollama",
                "ollama_base_url": "http://localhost:11434",
                "ollama_model": "qwen3:8b",
                "openai_api_key": "",
                "openai_base_url": "https://api.openai.com/v1",
                "openai_model": "gpt-4o-mini",
                "aliyun_api_key": "",
                "aliyun_model": "qwen-turbo",
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 60,
            },
            "parser": {
                "auto_detect": True,
                "txt_chapter_pattern": "^(第.{1,10}章|Chapter \\d+)",
            },
            "storage": {
                "data_dir": str(ConfigManager.DEFAULT_CONFIG_DIR / "data"),
                "vector_db_type": "chroma",
                "graph_db_type": "json",
                "cache_enabled": True,
                "analysis_enabled": True,
                "analysis_max_chars": 8000,
                "features_enabled": False,
                "embedding_enabled": False,
                "embedding_max_chars": 4000,
                "embed_summary": True,
                "embed_chapters": False,
                "paragraph_enabled": True,
                "paragraph_min_chars": 60,
                "paragraph_semantic_enabled": False,
            },
            "log_level": "INFO",
        }
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(default_config, handle, sort_keys=False)
        return path

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> KnowledgeBaseConfig:
        path = Path(config_path) if config_path else Path(
            os.environ.get("NOVEL_KB_CONFIG", ConfigManager.DEFAULT_CONFIG_FILE)
        )
        if not path.exists():
            ConfigManager.ensure_default(str(path))
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        llm = LLMConfig(**data.get("llm", {}))
        parser = ParserConfig(**data.get("parser", {}))
        storage_data = data.get("storage", {})
        storage_dir = Path(storage_data.get("data_dir", ConfigManager.DEFAULT_CONFIG_DIR / "data")).expanduser()
        storage = StorageConfig(
            data_dir=storage_dir,
            vector_db_type=storage_data.get("vector_db_type", "chroma"),
            graph_db_type=storage_data.get("graph_db_type", "json"),
            cache_enabled=storage_data.get("cache_enabled", True),
            analysis_enabled=storage_data.get("analysis_enabled", True),
            analysis_max_chars=storage_data.get("analysis_max_chars", 8000),
            features_enabled=storage_data.get("features_enabled", False),
            embedding_enabled=storage_data.get("embedding_enabled", False),
            embedding_max_chars=storage_data.get("embedding_max_chars", 4000),
            embed_summary=storage_data.get("embed_summary", True),
            embed_chapters=storage_data.get("embed_chapters", False),
            paragraph_enabled=storage_data.get("paragraph_enabled", True),
            paragraph_min_chars=storage_data.get("paragraph_min_chars", 60),
            paragraph_semantic_enabled=storage_data.get("paragraph_semantic_enabled", False),
        )
        return KnowledgeBaseConfig(
            llm=llm,
            parser=parser,
            storage=storage,
            log_level=data.get("log_level", "INFO"),
        )
