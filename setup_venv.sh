#!/bin/bash
# 设置虚拟环境和安装依赖

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== 设置 Novel Knowledge Base 虚拟环境 ==="
echo ""

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "虚拟环境创建完成"
else
    echo "虚拟环境已存在"
fi
echo ""

# 激活虚拟环境并安装依赖
echo "激活虚拟环境并安装依赖..."
source venv/bin/activate

echo "安装 Python 依赖..."
pip install --upgrade pip
pip install pyyaml requests aiohttp mcp openai volcengine-ark-runtime

echo ""
echo "=== 设置完成 ==="
echo ""
echo "现在可以运行: ./start_mcp_server.sh"