import os
from pathlib import Path
from typing import Optional

import yaml

from novel_kb.config.config_schema import KnowledgeBaseConfig, LLMConfig, ParserConfig, RootUserConfig, StorageConfig
from novel_kb.utils.logger import logger


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
                "provider": "xunfei",
                # 新格式：模型列表
                "xunfei_models": ["xopglm5-fast", "xopglm5"],
                "qwen_models": ["qwen-turbo", "qwen-plus"],
                "openai_models": ["gpt-4o-mini", "gpt-4"],
                "ollama_models": ["qwen3:8b"],
                "volcengine_models": ["doubao-1-5-pro-32k-250115"],
                "openai_like": {
                    "siliconflow": {
                        "api_key": "",
                        "base_url": "https://api.siliconflow.cn/v1",
                        "models": ["Qwen/Qwen2.5-7B-Instruct"],
                    }
                },
                # 高级模型标记
                "advanced_models": {
                    "xunfei:advanced": "xopglm5",
                    "qwen:advanced": "qwen-plus",
                },
                # Provider 配置
                "xunfei_api_key": "",
                "xunfei_base_url": "http://maas-api.cn-huabei-1.xf-yun.com/v2",
                "xunfei_lora_id": "",
                "xunfei_embedding_model": "",
                "aliyun_base_url": "https://dashscope.aliyuncs.com/compatible/openai/v1",
                "aliyun_api_key": "",
                "openai_api_key": "",
                "openai_base_url": "https://api.openai.com/v1",
                "ollama_base_url": "http://localhost:11434",
                "volcengine_api_key": "",
                "volcengine_base_url": "https://ark.cn-beijing.volces.com/api/v3",
                # 通用参数
                "temperature": 0.7,
                "max_tokens": [1024, 2048],  # [快速型, 高级型]
                "summary_max_tokens": 4096,
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
                "hierarchical_analysis_enabled": True,
                "segment_enabled": True,
                "segment_min_chars": 120,
                "segment_max_chars": 0,
                "segment_concurrency": 10,
                "segment_qps": 5.0,
                "segment_retries": 3,
                "segment_retry_interval": 1.0,
                "analysis_max_chars": 8000,
                "characters_enabled": False,
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
        logger.info("Loading config from: %s", path)
        if not path.exists():
            ConfigManager.ensure_default(str(path))
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        
        # 处理 LLM 配置中的旧格式兼容性
        llm_data = data.get("llm", {})
        llm_data = ConfigManager._migrate_llm_config(llm_data)
        
        llm = LLMConfig(**llm_data)
        parser = ParserConfig(**data.get("parser", {}))
        storage_data = data.get("storage", {})
        storage_dir = Path(storage_data.get("data_dir", ConfigManager.DEFAULT_CONFIG_DIR / "data")).expanduser()
        storage = StorageConfig(
            data_dir=storage_dir,
            vector_db_type=storage_data.get("vector_db_type", "chroma"),
            graph_db_type=storage_data.get("graph_db_type", "json"),
            cache_enabled=storage_data.get("cache_enabled", True),
            analysis_enabled=storage_data.get("analysis_enabled", True),
            hierarchical_analysis_enabled=storage_data.get("hierarchical_analysis_enabled", False),
            segment_enabled=storage_data.get("segment_enabled", True),
            segment_min_chars=storage_data.get("segment_min_chars", 120),
            segment_max_chars=storage_data.get("segment_max_chars", 0),
            segment_concurrency=storage_data.get("segment_concurrency", 2),
            segment_qps=storage_data.get("segment_qps", 1.0),
            segment_retries=storage_data.get("segment_retries", 3),
            segment_retry_interval=storage_data.get("segment_retry_interval", 1.0),
            analysis_max_chars=storage_data.get("analysis_max_chars", 8000),
            characters_enabled=storage_data.get("characters_enabled", True),
            features_enabled=storage_data.get("features_enabled", False),
            embedding_enabled=storage_data.get("embedding_enabled", False),
            embedding_max_chars=storage_data.get("embedding_max_chars", 4000),
            embed_summary=storage_data.get("embed_summary", True),
            embed_chapters=storage_data.get("embed_chapters", False),
            paragraph_enabled=storage_data.get("paragraph_enabled", True),
            paragraph_min_chars=storage_data.get("paragraph_min_chars", 60),
            paragraph_semantic_enabled=storage_data.get("paragraph_semantic_enabled", False),
        )
        root_user_data = data.get("root_user", {})
        root_user = RootUserConfig(
            name=root_user_data.get("name", "admin"),
            api_key=root_user_data.get("api_key", ""),
        )
        return KnowledgeBaseConfig(
            llm=llm,
            parser=parser,
            storage=storage,
            log_level=data.get("log_level", "INFO"),
            root_user=root_user,
        )

    @staticmethod
    def _migrate_llm_config(llm_data: dict) -> dict:
        """
        处理 LLM 配置的格式迁移
        支持两种格式：
        - 新格式: xxx_models: [model1, model2]
        - 旧格式: xxx_model: model_name（自动转换为列表）
        """
        result = dict(llm_data)
        
        # 如果使用旧格式（xxx_model），转换为新格式（xxx_models）
        for provider in ["ollama", "openai", "aliyun", "xunfei", "volcengine"]:
            old_key = f"{provider}_model"
            new_key = f"{provider}_models"
            
            # 如果旧格式存在
            if old_key in result:
                model = result[old_key]
                # 如果新格式不存在，用旧格式的值创建新格式
                if new_key not in result and model:
                    result[new_key] = [model] if isinstance(model, str) else model
                # 移除旧 key，避免传递给 LLMConfig
                del result[old_key]
        
        # 确保所有 xxx_models 都存在（有默认值）
        defaults = {
            "ollama_models": ["mistral:7b"],
            "openai_models": ["gpt-4o-mini"],
            "aliyun_models": ["qwen-turbo"],
            "xunfei_models": ["xopglm5"],
            "volcengine_models": ["doubao-1-5-pro-32k-250115"],
        }
        for key, default_value in defaults.items():
            if key not in result or not result[key]:
                result[key] = default_value

        # 规范化 openai_like
        openai_like = result.get("openai_like")
        if not isinstance(openai_like, dict):
            result["openai_like"] = {}
            return result

        normalized_openai_like = {}
        for provider, provider_cfg in openai_like.items():
            if not isinstance(provider, str) or not provider.strip():
                continue
            if not isinstance(provider_cfg, dict):
                continue

            cfg = dict(provider_cfg)
            models = cfg.get("models")
            if isinstance(models, str) and models:
                cfg["models"] = [models]
            elif isinstance(models, list):
                cfg["models"] = [m for m in models if isinstance(m, str) and m]
            else:
                cfg["models"] = []

            normalized_openai_like[provider.strip()] = cfg

        result["openai_like"] = normalized_openai_like
        
        return result
