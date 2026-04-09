import copy
from typing import Any, Dict, Literal, Optional

from novel_kb.llm.ollama_provider import OllamaProvider
from novel_kb.llm.openai_like_provider import OpenAILikeProvider
from novel_kb.llm.provider import LLMProvider
from novel_kb.utils.logger import logger


class LLMFactory:
    _providers = {
        "ollama": OllamaProvider,
    }

    _openai_compatible_builtin = {"openai", "aliyun", "xunfei", "volcengine"}

    @classmethod
    def create(
        cls,
        config,
        model_selection: Literal["fast", "advanced", "default"] = "fast",
        provider_type: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> LLMProvider:
        """
        创建 LLM provider

        Args:
            config: LLMConfig
            model_selection: 模型选择
                - "fast": 快速型模型（第一个）
                - "advanced": 高级型模型（最后一个或标记的）
                - "default": 默认（当前 provider 配置的）
            provider_type: 指定要创建的 provider 类型（用于多 provider 场景）
                           如果为 None，则使用 config.provider（可能是字符串或列表）
        """
        # 如果启用 gateway 模式，创建 gateway client
        if getattr(config, "use_gateway", False):
            from novel_kb.gateway_client import GatewayClient
            gateway_url = getattr(config, "gateway_url", "http://127.0.0.1:8747")
            gateway_tier = getattr(config, "gateway_tier", "medium")
            logger.info("Creating Gateway client: url=%s tier=%s", gateway_url, gateway_tier)
            return GatewayClient(base_url=gateway_url, tier=gateway_tier)

        # 如果没有指定 provider_type，使用配置中的（支持单个或列表）
        if provider_type is None:
            if isinstance(config.provider, list):
                provider_type = config.provider[0] if config.provider else "ollama"
            else:
                provider_type = config.provider
        
        # 应用模型选择逻辑
        config_copy = cls._apply_model_selection(config, model_selection, provider_type)
        # 应用模型覆盖（用于多模型轮询）
        if model_override:
            config_copy = cls._apply_model_override(config_copy, provider_type, model_override)

        provider_class = cls._providers.get(provider_type)
        if provider_class is not None:
            provider = provider_class(config_copy)
        else:
            provider = cls._create_openai_like_provider(config_copy, provider_type)

        # Attach provider metadata for logging/debugging
        provider.provider_type = provider_type
        provider.model_name = model_override or cls._get_effective_model(config_copy, provider_type)
        logger.info(
            "LLM provider created: %s model=%s",
            provider_type,
            cls._get_effective_model(config_copy, provider_type),
        )
        return provider

    @classmethod
    def _apply_model_selection(cls, config, model_selection: str, provider_type: str):
        """根据模型选择策略调整配置"""
        if model_selection == "default":
            return config

        mapper = config.get_model_mapper()

        if model_selection == "fast":
            selected_model = mapper.get_fast_model(provider_type)
        elif model_selection == "advanced":
            selected_model = mapper.get_advanced_model(provider_type)
        else:
            selected_model = None

        if not selected_model:
            return config

        config_copy = copy.copy(config)
        if hasattr(config_copy, "set_provider_model"):
            config_copy.set_provider_model(provider_type, selected_model)

        return config_copy

    @classmethod
    def _apply_model_override(cls, config, provider_type: str, model_override: str):
        """强制指定模型并调整 max_tokens"""
        config_copy = copy.copy(config)
        if hasattr(config_copy, "set_provider_model"):
            config_copy.set_provider_model(provider_type, model_override)

        if hasattr(config_copy, "get_max_tokens_for_provider_model"):
            config_copy.max_tokens = config_copy.get_max_tokens_for_provider_model(
                provider_type,
                model_override,
            )

        return config_copy

    @staticmethod
    def _get_effective_model(config, provider_type: str) -> Optional[str]:
        """获取配置中实际使用的模型"""
        if hasattr(config, "get_effective_model"):
            return config.get_effective_model(provider_type)
        if provider_type == "ollama":
            return getattr(config, "ollama_model", None)
        return None

    @classmethod
    def _create_openai_like_provider(cls, config, provider_type: str) -> LLMProvider:
        provider_cfg = cls._resolve_openai_like_config(config, provider_type)
        if not provider_cfg:
            raise ValueError(
                f"Unknown provider: {provider_type}. No dedicated provider block found, "
                f"and llm.openai_like does not contain this provider"
            )

        model = cls._get_effective_model(config, provider_type)
        if not model:
            models = provider_cfg.get("models")
            if isinstance(models, list) and models:
                first = models[0]
                model = first if isinstance(first, str) and first else None
            elif isinstance(models, str) and models:
                model = models
        if not model:
            raise ValueError(f"Provider {provider_type} has no model configured")

        max_tokens = config.get_max_tokens_for_provider_model(provider_type, model)
        return OpenAILikeProvider(
            provider_name=provider_type,
            model=model,
            api_key=str(provider_cfg.get("api_key", "")),
            base_url=str(provider_cfg.get("base_url", "")).rstrip("/"),
            temperature=float(getattr(config, "temperature", 0.7)),
            max_tokens=int(max_tokens),
            timeout=int(getattr(config, "timeout", 60)),
            embedding_model=str(provider_cfg.get("embedding_model", "")),
            extra_headers=cls._dict_str_or_none(provider_cfg.get("extra_headers")),
            extra_body=cls._dict_or_none(provider_cfg.get("extra_body")),
        )

    @classmethod
    def _resolve_openai_like_config(cls, config, provider_type: str) -> Optional[Dict[str, Any]]:
        if provider_type in cls._openai_compatible_builtin:
            builtin = cls._resolve_builtin_openai_like_config(config, provider_type)
            if builtin:
                return builtin

        if hasattr(config, "get_openai_like_provider_config"):
            custom_cfg = config.get_openai_like_provider_config(provider_type)
            if custom_cfg:
                return custom_cfg
        return None

    @staticmethod
    def _resolve_builtin_openai_like_config(config, provider_type: str) -> Optional[Dict[str, Any]]:
        if provider_type == "openai":
            return {
                "api_key": getattr(config, "openai_api_key", ""),
                "base_url": getattr(config, "openai_base_url", "https://api.openai.com/v1"),
                "models": getattr(config, "openai_models", []),
            }
        if provider_type == "aliyun":
            return {
                "api_key": getattr(config, "aliyun_api_key", ""),
                "base_url": getattr(config, "aliyun_base_url", "https://dashscope.aliyuncs.com/compatible/openai/v1"),
                "models": getattr(config, "aliyun_models", []),
            }
        if provider_type == "xunfei":
            headers = None
            lora_id = getattr(config, "xunfei_lora_id", "")
            if lora_id:
                headers = {"lora_id": lora_id}
            return {
                "api_key": getattr(config, "xunfei_api_key", ""),
                "base_url": getattr(config, "xunfei_base_url", "http://maas-api.cn-huabei-1.xf-yun.com/v2"),
                "models": getattr(config, "xunfei_models", []),
                "embedding_model": getattr(config, "xunfei_embedding_model", ""),
                "extra_headers": headers,
                "extra_body": {
                    "response_format": {"type": "json_object"},
                    "search_disable": True,
                },
            }
        if provider_type == "volcengine":
            return {
                "api_key": getattr(config, "volcengine_api_key", ""),
                "base_url": getattr(config, "volcengine_base_url", "https://ark.cn-beijing.volces.com/api/v3"),
                "models": getattr(config, "volcengine_models", []),
            }
        return None

    @staticmethod
    def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
        return value if isinstance(value, dict) else None

    @staticmethod
    def _dict_str_or_none(value: Any) -> Optional[Dict[str, str]]:
        if not isinstance(value, dict):
            return None
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if isinstance(item, str):
                result[key] = item
        return result or None

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        cls._providers[name] = provider_class
