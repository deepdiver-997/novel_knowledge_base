from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class LLMConfig:
    provider: Literal["ollama", "openai", "aliyun"]
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    aliyun_api_key: str = ""
    aliyun_model: str = "qwen-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class ParserConfig:
    auto_detect: bool = True
    txt_chapter_pattern: str = r"^(第.{1,10}章|Chapter \\d+)"


@dataclass
class StorageConfig:
    data_dir: Path
    vector_db_type: Literal["chroma"] = "chroma"
    graph_db_type: Literal["json", "sqlite"] = "json"
    cache_enabled: bool = True
    analysis_enabled: bool = True
    analysis_max_chars: int = 8000
    features_enabled: bool = False
    embedding_enabled: bool = False
    embedding_max_chars: int = 4000
    embed_summary: bool = True
    embed_chapters: bool = False
    paragraph_enabled: bool = True
    paragraph_min_chars: int = 60
    paragraph_semantic_enabled: bool = False


@dataclass
class KnowledgeBaseConfig:
    llm: LLMConfig
    parser: ParserConfig
    storage: StorageConfig
    log_level: str = "INFO"
