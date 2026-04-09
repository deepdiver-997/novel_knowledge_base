#!/usr/bin/env bash
# Start/Stop the LLM Gateway server

set -euo pipefail

GATEWAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../gateway && pwd)"
PID_FILE="/tmp/gateway_8747.pid"

start_gateway() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Gateway already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi

    echo "Starting LLM Gateway..."
    cd "$GATEWAY_DIR"
    PYTHONPATH="$GATEWAY_DIR" python main.py > /tmp/gateway.log 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3

    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Gateway started (PID: $(cat "$PID_FILE"))"
        # Wait for server to be ready
        for i in 1 2 3 4 5; do
            if curl -s --max-time 3 http://127.0.0.1:8747/v1/health > /dev/null 2>&1; then
                echo "Gateway is ready"
                return 0
            fi
            sleep 1
        done
        echo "Gateway may still be starting (check status)"
    else
        echo "Failed to start gateway. Check /tmp/gateway.log"
        return 1
    fi
}

stop_gateway() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping gateway (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            sleep 1
            echo "Gateway stopped"
        else
            echo "Gateway not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        echo "Gateway not running"
    fi
}

status_gateway() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Gateway running (PID: $(cat "$PID_FILE"))"
        curl -s --max-time 3 http://127.0.0.1:8747/v1/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Providers:', list(d.get('result',{}).get('providers',{}).keys()))" 2>/dev/null || true
    else
        echo "Gateway not running"
    fi
}

case "${1:-start}" in
    start)
        start_gateway
        ;;
    stop)
        stop_gateway
        ;;
    restart)
        stop_gateway
        sleep 1
        start_gateway
        ;;
    status)
        status_gateway
        ;;
    log)
        tail -20 /tmp/gateway.log
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
