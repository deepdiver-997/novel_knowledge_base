#!/usr/bin/env python3
"""快速测试MCP服务器的基本功能"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    try:
        from novel_kb.mcp.server import NovelKBMCPServer
        print("✓ MCP服务器模块导入成功")
    except Exception as e:
        print(f"✗ MCP服务器模块导入失败: {e}")
        return False

    try:
        from novel_kb.config.config_manager import ConfigManager
        print("✓ 配置管理模块导入成功")
    except Exception as e:
        print(f"✗ 配置管理模块导入失败: {e}")
        return False

    try:
        from novel_kb.services.search_service import SearchService
        print("✓ 搜索服务模块导入成功")
    except Exception as e:
        print(f"✗ 搜索服务模块导入失败: {e}")
        return False

    return True

def test_config():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    try:
        from novel_kb.config.config_manager import ConfigManager
        config = ConfigManager.load_config()
        print(f"✓ 配置加载成功")
        print(f"  数据目录: {config.storage.data_dir}")
        print(f"  LLM提供商: {config.llm.get_providers()}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False

def test_tools():
    """测试工具处理器"""
    print("\n=== 测试工具处理器 ===")
    try:
        from novel_kb.mcp.handlers.tool_handler import ToolHandler
        from novel_kb.config.config_manager import ConfigManager
        
        config = ConfigManager.load_config()
        handler = ToolHandler(config)
        
        print(f"✓ 工具处理器创建成功")
        print(f"  可用工具数量: {len(handler.tools)}")
        
        # 列出一些关键工具
        tool_names = [tool.name for tool in handler.tools]
        key_tools = ['hierarchical_search', 'search_novel', 'search_chapters', 'get_novel_hierarchy']
        
        for key_tool in key_tools:
            if key_tool in tool_names:
                print(f"  ✓ 工具 '{key_tool}' 可用")
            else:
                print(f"  ✗ 工具 '{key_tool}' 不可用")
        
        return True
    except Exception as e:
        print(f"✗ 工具处理器创建失败: {e}")
        return False

def test_data():
    """测试数据访问"""
    print("\n=== 测试数据访问 ===")
    try:
        from novel_kb.knowledge_base.repository import NovelRepository
        from novel_kb.config.config_manager import ConfigManager
        
        config = ConfigManager.load_config()
        repo = NovelRepository(config.storage.data_dir)
        
        novels = repo.list_novels()
        print(f"✓ 数据仓库访问成功")
        print(f"  已导入小说数量: {len(novels)}")
        
        if novels:
            print(f"  示例小说:")
            for novel in novels[:3]:
                print(f"    - {novel.novel_id}: {novel.title}")
        
        return True
    except Exception as e:
        print(f"✗ 数据仓库访问失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("Novel Knowledge Base MCP 服务器测试")
    print("=" * 50)
    
    results = []
    
    results.append(("模块导入", test_imports()))
    results.append(("配置加载", test_config()))
    results.append(("工具处理器", test_tools()))
    results.append(("数据访问", test_data()))
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ 所有测试通过！MCP服务器可以正常启动。")
        print("运行以下命令启动服务器:")
        print("  ./start_mcp_server.sh")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())