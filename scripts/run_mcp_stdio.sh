#!/usr/bin/env bash
set -euo pipefail

# Portable runner to start an MCP stdio server for this repository.
# Usage:
#   sh /path/to/scripts/run_mcp_stdio.sh [--help] [extra args]
# Environment overrides:
#   MCP_MODULE : Python module to run with -m (default: novel_kb.mcp.server)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer project virtualenvs
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY_EXEC="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/venv/bin/python" ]; then
  PY_EXEC="$REPO_ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_EXEC="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY_EXEC="$(command -v python)"
else
  echo "Error: No suitable Python interpreter found." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MCP_MODULE="${MCP_MODULE:-novel_kb.mcp.server}"

echo "Repository root: $REPO_ROOT" >&2
echo "Python: $PY_EXEC" >&2
echo "MCP module: $MCP_MODULE" >&2

# Forward any additional args to the Python module
exec "$PY_EXEC" -m "$MCP_MODULE" "$@"
