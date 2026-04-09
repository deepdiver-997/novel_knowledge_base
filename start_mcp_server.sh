#!/usr/bin/env bash
set -euo pipefail
# Robust MCP startup script

# Resolve script directory and switch to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Prefer project virtualenvs: .venv -> venv -> $VIRTUAL_ENV -> system python
PYTHON_CMD=""
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
elif [ -d "venv" ] && [ -x "venv/bin/python" ]; then
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
else
    echo "ERROR: No suitable Python interpreter found. Create a virtualenv (./setup_venv.sh)" >&2
    exit 1
fi

# Export minimal env to ensure project imports work
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NOVEL_KB_LOG_LEVEL="${NOVEL_KB_LOG_LEVEL:-INFO}"

CONFIG_FILE="$HOME/.novel_knowledge_base/config.yaml"
echo "Starting Novel Knowledge Base MCP Server..."
echo "Project root: $SCRIPT_DIR"
echo "Using python: $PYTHON_CMD"
echo "Config file: $CONFIG_FILE"
echo "Log level: $NOVEL_KB_LOG_LEVEL"

# Use unbuffered mode so logs stream correctly
exec "$PYTHON_CMD" -u ./scripts/run_mcp_stdlib.py