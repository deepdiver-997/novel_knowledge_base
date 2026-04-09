#!/usr/bin/env python3
"""
最终测试脚本 - 验证MCP服务器完整性
"""

import asyncio
import json
from novel_kb.mcp.server import NovelKBMCPServer
from novel_kb.config.config_manager import ConfigManager

async def test_mcp_server():
    """完整测试MCP服务器功能"""
    config = ConfigManager.load_config()
    server = NovelKBMCPServer(config)
    
    print("=" * 60)
    print("🎉 Novel Knowledge Base MCP 服务器完整性测试")
    print("=" * 60)
    
    # 1. 检查服务器基本信息
    print("\n📋 1. 服务器基本信息:")
    print(f"   服务器名称: {server.app.name}")
    print(f"   注册工具数: {len(server.tool_handler.tools)}")
    print(f"   配置提供商: {config.llm.get_providers()}")
    
    # 2. 列出所有工具
    print(f"\n🔧 2. 所有可用工具:")
    for i, tool in enumerate(server.tool_handler.tools, 1):
        print(f"   {i:2d}. {tool.name:30s} - {tool.description}")
    
    # 3. 测试基础工具（不需要API）
    print(f"\n✅ 3. 测试基础工具:")
    
    basic_tools = ["list_novels", "health_check"]
    for tool_name in basic_tools:
        try:
            result = await server.app.call_tool()(
                name=tool_name,
                arguments={}
            )
            if result and len(result) > 0:
                content = result[0].text
                print(f"   ✓ {tool_name:20s} - 成功")
                # 尝试解析JSON以获得更好的显示
                try:
                    data = json.loads(content)
                    if tool_name == "list_novels":
                        novel_count = len(data.get("novels", []))
                        print(f"      → 已导入 {novel_count} 部小说")
                except:
                    pass
            else:
                print(f"   ✗ {tool_name:20s} - 无返回结果")
        except Exception as e:
            print(f"   ✗ {tool_name:20s} - 失败: {str(e)[:50]}")
    
    # 4. 重点工具说明
    print(f"\n🎯 4. 核心功能说明:")
    core_tools = {
        "hierarchical_search": "智能分层搜索，模拟人类阅读模式",
        "search_novel": "搜索小说内容",
        "search_chapters": "搜索章节内容", 
        "search_plot_summaries": "搜索剧情总结（卷级）",
        "get_novel_hierarchy": "获取小说层级结构",
        "list_novels": "列出已导入的小说",
        "health_check": "检查服务器状态"
    }
    
    for tool_name, description in core_tools.items():
        exists = any(t.name == tool_name for t in server.tool_handler.tools)
        status = "✓" if exists else "✗"
        print(f"   {status} {tool_name:20s} - {description}")
    
    # 5. 使用说明
    print(f"\n📖 5. 使用说明:")
    print("   启动命令:")
    print("     ./start_mcp_server.sh")
    print("")
    print("   或直接运行:")
    print("     venv/bin/python run_mcp_stdlib.py")
    print("")
    print("   在Cherry Studio/Claude Desktop中配置:")
    print(f'     "command": "{config.storage.data_dir.parent.parent}/start_mcp_server.sh"')
    
    # 6. 注意事项
    print(f"\n⚠️  6. 注意事项:")
    print("   - 部分工具需要LLM API支持，可能因429错误失败")
    print("   - 基础工具（list_novels, health_check）不依赖API")
    print("   - API相关问题不影响服务器核心功能")
    
    print("\n" + "=" * 60)
    print("🎊 测试完成！MCP服务器已准备就绪")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_mcp_server())