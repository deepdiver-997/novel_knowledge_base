import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TierConfig:
    provider: str
    model: str


@dataclass
class ProviderConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    embedding_model: str = ""
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None
    lora_id: str = ""


@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8747
    log_level: str = "INFO"

    # Tier -> (provider, model) mapping
    tiers: Dict[str, TierConfig] = field(default_factory=dict)

    # Provider configurations
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "GatewayConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get("GATEWAY_CONFIG", "~/.novel_knowledge_base/gateway_config.yaml")

        path = Path(config_path).expanduser()
        if not path.exists():
            return cls._default_config()

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _default_config(cls) -> "GatewayConfig":
        """Return a default configuration."""
        return cls(
            tiers={
                "low": TierConfig(provider="aliyun", model="qwen-turbo"),
                "medium": TierConfig(provider="aliyun", model="qwen-plus"),
                "high": TierConfig(provider="aliyun", model="qwen-max"),
            },
            providers={
                "aliyun": ProviderConfig(
                    api_key=os.environ.get("ALIYUN_API_KEY", ""),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model="qwen-turbo",
                ),
            },
        )

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        tiers: Dict[str, TierConfig] = {}
        tier_data = data.get("tiers", {})
        if isinstance(tier_data, dict):
            for name, cfg in tier_data.items():
                if isinstance(cfg, dict):
                    tiers[name] = TierConfig(
                        provider=cfg.get("provider", ""),
                        model=cfg.get("model", ""),
                    )

        providers: Dict[str, ProviderConfig] = {}
        provider_data = data.get("providers", {})
        if isinstance(provider_data, dict):
            for name, cfg in provider_data.items():
                if isinstance(cfg, dict):
                    providers[name] = ProviderConfig(
                        api_key=cfg.get("api_key", ""),
                        base_url=cfg.get("base_url", ""),
                        model=cfg.get("model", ""),
                        temperature=cfg.get("temperature", 0.7),
                        max_tokens=cfg.get("max_tokens", 2048),
                        timeout=cfg.get("timeout", 60),
                        embedding_model=cfg.get("embedding_model", ""),
                        extra_headers=cfg.get("extra_headers"),
                        extra_body=cfg.get("extra_body"),
                        lora_id=cfg.get("lora_id", ""),
                    )

        return cls(
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 8747),
            log_level=data.get("log_level", "INFO"),
            tiers=tiers,
            providers=providers,
        )
