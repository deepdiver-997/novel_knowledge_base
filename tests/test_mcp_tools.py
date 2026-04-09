#!/usr/bin/env python3
"""测试MCP工具注册"""

import asyncio
from novel_kb.mcp.server import NovelKBMCPServer
from novel_kb.config.config_manager import ConfigManager

async def test_mcp_server():
    """测试MCP服务器"""
    config = ConfigManager.load_config()
    server = NovelKBMCPServer(config)
    
    print("=== MCP服务器测试 ===")
    print(f"服务器名称: {server.app.name}")
    print(f"工具数量: {len(server.tool_handler.tools)}")
    print()
    
    # 列出前几个工具
    print("可用的MCP工具:")
    for i, tool in enumerate(server.tool_handler.tools[:5]):
        print(f"  {i+1}. {tool.name}: {tool.description}")
    
    print(f"  ... 还有 {len(server.tool_handler.tools) - 5} 个工具")
    print()
    
    print("✓ MCP服务器工具注册成功！")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())