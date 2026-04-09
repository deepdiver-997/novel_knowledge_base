import json
import sys
from typing import Any

from novel_kb.config.config_manager import ConfigManager
from novel_kb.llm.factory import LLMFactory


def _print_response(label: str, data: Any) -> None:
    print(f"\n=== {label} ===")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(data)


def main() -> int:
    config = ConfigManager.load_config(str(ConfigManager.DEFAULT_CONFIG_FILE))
    provider = LLMFactory.create(config.llm)

    print(f"provider={config.llm.provider}")
    print("health_check=", provider.health_check())

    try:
        result = provider.analyze_plot("这是一个简短的故事片段，用来测试接口是否可用。")
    except Exception as exc:
        print("API call failed:", exc)
        return 1

    _print_response("analyze_plot", result.content)
    print("tokens_used=", result.tokens_used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
