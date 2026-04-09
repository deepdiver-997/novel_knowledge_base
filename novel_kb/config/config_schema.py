from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, List, Dict, Optional, cast


def _empty_str_dict() -> Dict[str, str]:
    return {}


def _empty_openai_like_dict() -> Dict[str, Dict[str, Any]]:
    return {}


@dataclass
class ModelMapper:
    """模型映射管理器"""
    # Provider 类型 → 模型列表（[快速型, ..., 高级型]）
    models_by_provider: Dict[str, List[str]]
    # 高级模型标记: "provider:turbo" → model_name
    advanced_models: Dict[str, str]

    @classmethod
    def from_llm_config(cls, config: "LLMConfig") -> "ModelMapper":
        """从 LLMConfig 生成映射"""
        models: Dict[str, List[str]] = {}
        # 按 provider 分类模型
        if config.ollama_models:
            models["ollama"] = config.ollama_models
        if config.openai_models:
            models["openai"] = config.openai_models
        if config.aliyun_models:
            models["aliyun"] = config.aliyun_models
        if config.xunfei_models:
            models["xunfei"] = config.xunfei_models
        if config.volcengine_models:
            models["volcengine"] = config.volcengine_models
        if config.openai_like:
            for provider, provider_cfg in config.openai_like.items():
                provider_models = provider_cfg.get("models")
                if isinstance(provider_models, list):
                    normalized: List[str] = []
                    for item in cast(List[Any], provider_models):
                        if isinstance(item, str) and item:
                            normalized.append(item)
                    if normalized:
                        models[provider] = normalized
                elif isinstance(provider_models, str) and provider_models:
                    models[provider] = [provider_models]

        # 高级模型映射
        advanced = config.advanced_models or {}

        return cls(models_by_provider=models, advanced_models=advanced)

    def get_models(self, provider: str) -> List[str]:
        """获取某 provider 所有模型"""
        return self.models_by_provider.get(provider, [])

    def get_fast_model(self, provider: str) -> Optional[str]:
        """获取快速型模型（第一个）"""
        models = self.get_models(provider)
        return models[0] if models else None

    def get_advanced_model(self, provider: str) -> Optional[str]:
        """获取高级型模型（最后一个或标记的）"""
        # 优先返回标记的高级型
        advanced_key = f"{provider}:advanced"
        if advanced_key in self.advanced_models:
            return self.advanced_models[advanced_key]
        # 否则返回列表中最后一个
        models = self.get_models(provider)
        return models[-1] if len(models) > 1 else (models[0] if models else None)

    def is_advanced(self, provider: str, model: str) -> bool:
        """判断是否为高级模型"""
        advanced_key = f"{provider}:advanced"
        if advanced_key in self.advanced_models:
            return self.advanced_models[advanced_key] == model
        # 否则根据位置判断（最后一个为高级）
        models = self.get_models(provider)
        return model == models[-1] if len(models) > 1 else False


@dataclass
class LLMConfig:
    # provider 支持单个或列表，用于多 provider 轮询提高并发
    provider: Union[str, List[str]]

    # 模型配置（列表格式：[快速型, ..., 高级型]）
    ollama_models: List[str] = field(default_factory=lambda: ["mistral:7b"])
    openai_models: List[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    aliyun_models: List[str] = field(default_factory=lambda: ["qwen-turbo", "qwen-plus"])
    xunfei_models: List[str] = field(default_factory=lambda: ["xopglm5"])
    volcengine_models: List[str] = field(default_factory=lambda: ["doubao-1-5-pro-32k-250115"])

    # 通用 OpenAI 兼容厂商配置
    # openai_like:
    #   provider_name:
    #     api_key: "..."
    #     base_url: "https://..."
    #     models: ["model-a", "model-b"]
    #     embedding_model: "text-embedding-3-small"  # 可选
    #     extra_headers: {"k": "v"}                 # 可选
    #     extra_body: {"k": "v"}                    # 可选
    openai_like: Dict[str, Dict[str, Any]] = field(default_factory=_empty_openai_like_dict)

    # 高级模型显式标记（可选，自动推导为列表最后一个）
    # 格式: {"provider:advanced": "model_name", ...}
    advanced_models: Dict[str, str] = field(default_factory=_empty_str_dict)

    # Provider 特定配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = ""  # 独立的 embedding 模型，如 nomic-embed-text
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    aliyun_api_key: str = ""
    aliyun_base_url: str = "https://dashscope.aliyuncs.com/compatible/openai/v1"
    volcengine_api_key: str = ""
    volcengine_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    xunfei_api_key: str = ""
    xunfei_base_url: str = "http://maas-api.cn-huabei-1.xf-yun.com/v2"
    xunfei_lora_id: str = ""
    xunfei_embedding_model: str = ""

    # 通用参数
    temperature: float = 0.7
    # max_tokens 支持单值或列表（对应模型列表）
    max_tokens: Union[int, List[int]] = 2048
    summary_max_tokens: int = 4096
    timeout: int = 60

    # Gateway 配置（使用本地 LLM Gateway 服务）
    use_gateway: bool = False
    gateway_url: str = "http://127.0.0.1:8747"
    gateway_tier: str = "medium"  # low/medium/high
    gateway_max_tokens: Optional[int] = None
    gateway_temperature: Optional[float] = None
    gateway_tier_short: Optional[str] = None  # 按文本长度自动切换 tier
    gateway_tier_long: Optional[str] = None
    gateway_short_chars: int = 1000
    gateway_long_chars: int = 4000

    def __post_init__(self):
        """初始化后自动生成单数 model 属性（用于兼容性）"""
        # 为了兼容旧代码，自动生成 ollama_model、openai_model 等属性
        # 默认使用快速型（列表第一个）
        mapper = self.get_model_mapper()

        # 直接赋值（dataclass 允许动态添加属性）
        self.ollama_model = mapper.get_fast_model("ollama") or "mistral:7b"
        self.openai_model = mapper.get_fast_model("openai") or "gpt-4o-mini"
        self.aliyun_model = mapper.get_fast_model("aliyun") or "qwen-turbo"
        self.xunfei_model = mapper.get_fast_model("xunfei") or "xopglm5"
        self.volcengine_model = mapper.get_fast_model("volcengine") or "doubao-1-5-pro-32k-250115"
    
    def get_providers(self) -> List[str]:
        """获取规范化的 provider 列表"""
        if isinstance(self.provider, list):
            return self.provider
        return [self.provider]
    
    def get_primary_provider(self) -> str:
        """获取主 provider（用于兼容性）"""
        providers = self.get_providers()
        return providers[0] if providers else "ollama"

    def get_model_mapper(self) -> ModelMapper:
        """生成模型映射器"""
        return ModelMapper.from_llm_config(self)

    def get_models_for_provider(self, provider: str) -> List[str]:
        """获取指定 provider 的模型列表"""
        mapper = self.get_model_mapper()
        models = mapper.get_models(provider)
        if models:
            return models
        openai_like_cfg = self.get_openai_like_provider_config(provider)
        if not openai_like_cfg:
            return []
        candidate = openai_like_cfg.get("models")
        if isinstance(candidate, list):
            models: List[str] = []
            for item in cast(List[Any], candidate):
                if isinstance(item, str) and item:
                    models.append(item)
            return models
        if isinstance(candidate, str) and candidate:
            return [candidate]
        return []

    def get_openai_like_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        value = self.openai_like.get(provider)
        return value if isinstance(value, dict) else None

    def set_provider_model(self, provider: str, model: str) -> None:
        if provider == "ollama":
            self.ollama_model = model
            return
        if provider == "openai":
            self.openai_model = model
            return
        if provider == "aliyun":
            self.aliyun_model = model
            return
        if provider == "xunfei":
            self.xunfei_model = model
            return
        if provider == "volcengine":
            self.volcengine_model = model
            return
        cfg = self.openai_like.get(provider)
        if isinstance(cfg, dict):
            cfg["model"] = model

    def get_effective_model(self, provider: str) -> Optional[str]:
        if provider == "ollama":
            return getattr(self, "ollama_model", None)
        if provider == "openai":
            return getattr(self, "openai_model", None)
        if provider == "aliyun":
            return getattr(self, "aliyun_model", None)
        if provider == "xunfei":
            return getattr(self, "xunfei_model", None)
        if provider == "volcengine":
            return getattr(self, "volcengine_model", None)
        cfg = self.openai_like.get(provider)
        if not isinstance(cfg, dict):
            return None
        model = cfg.get("model")
        if isinstance(model, str) and model:
            return model
        models = cfg.get("models")
        if isinstance(models, list) and models:
            first = cast(List[Any], models)[0]
            return first if isinstance(first, str) and first else None
        if isinstance(models, str) and models:
            return models
        return None

    def get_current_models(self) -> List[str]:
        """获取当前 provider 的所有模型"""
        mapper = self.get_model_mapper()
        return mapper.get_models(self.get_primary_provider())

    def get_current_fast_model(self) -> Optional[str]:
        """获取当前 provider 的快速型模型"""
        mapper = self.get_model_mapper()
        return mapper.get_fast_model(self.get_primary_provider())

    def get_current_advanced_model(self) -> Optional[str]:
        """获取当前 provider 的高级型模型"""
        mapper = self.get_model_mapper()
        return mapper.get_advanced_model(self.get_primary_provider())

    def get_max_tokens_for_model(self, model: Optional[str] = None) -> int:
        """获取某模型的 max_tokens"""
        if isinstance(self.max_tokens, int):
            return self.max_tokens

        mapper = self.get_model_mapper()
        models = mapper.get_models(self.get_primary_provider())

        if not model or not models:
            # 默认返回最后一个（高级型）
            return self.max_tokens[-1] if self.max_tokens else 2048

        try:
            idx = models.index(model)
            return self.max_tokens[idx] if idx < len(self.max_tokens) else self.max_tokens[-1]
        except (ValueError, IndexError):
            return self.max_tokens[-1] if self.max_tokens else 2048

    def get_max_tokens_for_provider_model(self, provider: str, model: Optional[str] = None) -> int:
        """获取指定 provider/模型的 max_tokens"""
        if isinstance(self.max_tokens, int):
            return self.max_tokens

        models = self.get_models_for_provider(provider)
        if not model or not models:
            return self.max_tokens[-1] if self.max_tokens else 2048

        try:
            idx = models.index(model)
            return self.max_tokens[idx] if idx < len(self.max_tokens) else self.max_tokens[-1]
        except (ValueError, IndexError):
            return self.max_tokens[-1] if self.max_tokens else 2048


@dataclass
class ParserConfig:
    auto_detect: bool = True
    txt_chapter_pattern: str = r"^(第.{1,10}章|Chapter \\d+)"


@dataclass
class StorageConfig:
    data_dir: Path
    vector_db_type: Literal["chroma", "memory"] = "chroma"
    graph_db_type: Literal["json", "sqlite"] = "json"
    cache_enabled: bool = True
    analysis_enabled: bool = True
    hierarchical_analysis_enabled: bool = True
    segment_enabled: bool = True
    segment_min_chars: int = 120
    segment_max_chars: int = 0
    segment_concurrency: int = 2
    segment_qps: float = 1.0
    segment_retries: int = 3
    segment_retry_interval: float = 1.0
    analysis_max_chars: int = 8000
    characters_enabled: bool = True
    features_enabled: bool = False

    # 向量存储配置
    embedding_enabled: bool = False
    embedding_max_chars: int = 4000
    embed_summary: bool = True
    embed_chapters: bool = False       # 原文嵌入，默认关闭（太大）
    embed_plot_summaries: bool = True  # 剧情总结嵌入
    embed_paragraphs: bool = False    # 原文分段嵌入，默认关闭
    paragraph_min_chars: int = 100
    paragraph_max_chars: int = 500
    paragraph_enabled: bool = True
    paragraph_semantic_enabled: bool = False


@dataclass
class RootUserConfig:
    name: str = "admin"
    api_key: str = ""


def _default_root_user() -> RootUserConfig:
    return RootUserConfig()


@dataclass
class KnowledgeBaseConfig:
    llm: LLMConfig
    parser: ParserConfig
    storage: StorageConfig
    log_level: str = "INFO"
    root_user: RootUserConfig = field(default_factory=_default_root_user)
