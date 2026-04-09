#!/bin/bash
# 测试 MCP 服务器启动脚本

echo "=== 测试 Novel Knowledge Base MCP 服务器 ==="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "当前目录: $(pwd)"
echo ""

# 检查Python
echo "检查 Python 环境:"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "  找到: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "  找到: $(python --version)"
else
    echo "  错误: 未找到 python 或 python3"
    exit 1
fi
echo ""

# 检查必要的Python包
echo "检查必要的Python包:"
"$PYTHON_CMD" -c "import yaml; print('  pyyaml: OK')" 2>/dev/null || echo "  pyyaml: 未安装"
"$PYTHON_CMD" -c "import mcp; print('  mcp: OK')" 2>/dev/null || echo "  mcp: 未安装"
"$PYTHON_CMD" -c "import aiohttp; print('  aiohttp: OK')" 2>/dev/null || echo "  aiohttp: 未安装"
echo ""

# 检查配置文件
CONFIG_FILE="$HOME/.novel_knowledge_base/config.yaml"
echo "检查配置文件:"
if [ -f "$CONFIG_FILE" ]; then
    echo "  配置文件存在: $CONFIG_FILE"
else
    echo "  警告: 配置文件不存在: $CONFIG_FILE"
    echo "  运行以下命令创建: $PYTHON_CMD -m novel_kb.main --init"
fi
echo ""

# 检查数据目录
DATA_DIR="$HOME/.novel_knowledge_base/data"
echo "检查数据目录:"
if [ -d "$DATA_DIR" ]; then
    echo "  数据目录存在: $DATA_DIR"
    NOVEL_COUNT=$(find "$DATA_DIR" -name "*.json" 2>/dev/null | wc -l)
    echo "  已导入的小说数量: $NOVEL_COUNT"
else
    echo "  数据目录不存在: $DATA_DIR"
fi
echo ""

# 测试导入核心模块
echo "测试导入核心模块:"
"$PYTHON_CMD" -c "from novel_kb.mcp.server import NovelKBMCPServer; print('  MCP服务器模块: OK')" 2>/dev/null || echo "  MCP服务器模块: 失败"
"$PYTHON_CMD" -c "from novel_kb.config.config_manager import ConfigManager; print('  配置管理模块: OK')" 2>/dev/null || echo "  配置管理模块: 失败"
echo ""

echo "=== 测试完成 ==="
echo ""
echo "如果所有检查都通过，可以运行以下命令启动服务器："
echo "  ./start_mcp_server.sh"