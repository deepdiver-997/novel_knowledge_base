from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NovelKBError(Exception):
    pass


class ConfigError(NovelKBError):
    pass


class ProviderError(NovelKBError):
    pass


class ProviderFailureAction(str, Enum):
    NONE = "none"
    THROTTLE = "throttle"
    DISABLE = "disable"


@dataclass(frozen=True)
class ProviderErrorDecision:
    action: ProviderFailureAction
    reason: Optional[str] = None
