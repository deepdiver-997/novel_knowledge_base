#!/usr/bin/env python3
"""测试MCP服务器基本功能（不依赖API）"""

import asyncio
from novel_kb.mcp.server import NovelKBMCPServer
from novel_kb.config.config_manager import ConfigManager

async def test_basic_functions():
    """测试不需要API的基本功能"""
    config = ConfigManager.load_config()
    server = NovelKBMCPServer(config)
    
    print("=== 测试不需要API的MCP功能 ===")
    
    # 测试 list_novels
    try:
        result = await server.app.call_tool()("list_novels", {})
        print("✓ list_novels 工具测试成功")
        if result and len(result) > 0:
            content = result[0]
            print(f"  返回内容预览: {str(content)[:100]}...")
    except Exception as e:
        print(f"✗ list_novels 测试失败: {e}")
    
    # 测试 health_check
    try:
        result = await server.app.call_tool()("health_check", {})
        print("✓ health_check 工具测试成功")
        if result and len(result) > 0:
            content = result[0]
            print(f"  返回内容预览: {str(content)[:100]}...")
    except Exception as e:
        print(f"✗ health_check 测试失败: {e}")
    
    print("\n=== 基础功能测试完成 ===")
    print("注意: 涉及LLM API的工具可能会因429错误而失败，这是正常的API限制。")

if __name__ == "__main__":
    asyncio.run(test_basic_functions())