#!/usr/bin/env python3
"""
独立MCP stdio启动脚本
直接通过stdio与LLM交互，无需集成到其他MCP

使用方法：
1. ./start_mcp_server.sh
2. 或者直接: python3 run_mcp_stdlib.py
3. 或者: python run_mcp_stdlib.py

LLM可以通过MCP stdio协议直接连接此服务
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from novel_kb.config.config_manager import ConfigManager
from novel_kb.mcp.server import NovelKBMCPServer
from novel_kb.utils.logger import logger


def main():
    """启动MCP stdio服务器"""
    try:
        # 加载配置
        config_path = os.environ.get("NOVEL_KB_CONFIG", None)
        config = ConfigManager.load_config(config_path)
        
        logger.info("Starting Novel Knowledge Base MCP Server...")
        logger.info(f"Data directory: {config.storage.data_dir}")
        logger.info(f"LLM provider: {config.llm.get_providers()}")
        
        # 创建并启动服务器
        server = NovelKBMCPServer(config)
        server.run()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()