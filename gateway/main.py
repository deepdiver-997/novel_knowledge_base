import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.config import GatewayConfig
from gateway.routes import Routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GatewayServer:
    """HTTP server for the LLM Gateway."""

    def __init__(self, config: Optional[GatewayConfig] = None) -> None:
        self.config = config or GatewayConfig.load()
        self.routes = Routes(self.config)
        self._server: Optional[asyncio.Server] = None

    async def handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming HTTP request."""
        addr = writer.get_extra_info("peername")
        logger.info(f"Connection from {addr}")

        try:
            # Read request line
            line = await reader.readline()
            if not line:
                await self._write_response(writer, 400, {"error": "Empty request"})
                return

            line = line.decode("utf-8").strip()
            if not line:
                await self._write_response(writer, 400, {"error": "Invalid request"})
                return

            parts = line.split(" ")
            if len(parts) < 2:
                await self._write_response(writer, 400, {"error": "Invalid request line"})
                return

            method, path = parts[0], parts[1]

            # Read headers
            headers: Dict[str, str] = {}
            content_length = 0
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n":
                    break
                line_str = line.decode("utf-8").strip()
                if ":" in line_str:
                    key, value = line_str.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
                    if key.strip().lower() == "content-length":
                        content_length = int(value.strip())

            # Read body if present
            body = b""
            if content_length > 0:
                body = await reader.readexactly(content_length)

            # Route request
            response = await self._route_request(method, path, headers, body)
            await self._write_response(writer, response.get("status", 200), response.get("body"))

        except Exception as exc:
            logger.exception(f"Error handling request from {addr}")
            await self._write_response(writer, 500, {"error": str(exc)})
        finally:
            writer.close()
            await writer.wait_closed()

    async def _route_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        # Health check: GET /v1/health
        if method == "GET" and path == "/v1/health":
            result = await self.routes.handle_health()
            return {
                "status": 200 if result.success else 500,
                "body": result.__dict__,
            }

        # Analyze: POST /v1/analyze
        if method == "POST" and path == "/v1/analyze":
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                return {"status": 400, "body": {"error": "Invalid JSON"}}

            method_name = data.get("method", "analyze_plot")
            tier = data.get("tier", "medium")
            text = data.get("text", "")

            if not text:
                return {"status": 400, "body": {"error": "text is required"}}

            result = await self.routes.handle_analyze(method_name, tier, text)
            return {
                "status": 200 if result.success else 500,
                "body": result.__dict__,
            }

        # Embed: POST /v1/embed
        if method == "POST" and path == "/v1/embed":
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                return {"status": 400, "body": {"error": "Invalid JSON"}}

            tier = data.get("tier", "medium")
            text = data.get("text", "")

            if not text:
                return {"status": 400, "body": {"error": "text is required"}}

            result = await self.routes.handle_embed(tier, text)
            return {
                "status": 200 if result.success else 500,
                "body": result.__dict__,
            }

        # 404 for unknown paths
        return {"status": 404, "body": {"error": f"Unknown path: {path}"}}

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: Optional[Dict[str, Any]],
    ) -> None:
        """Write HTTP response."""
        body_str = json.dumps(body, ensure_ascii=False) if body else ""
        body_bytes = body_str.encode("utf-8")

        response = (
            f"HTTP/1.1 {status} OK\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(response.encode("utf-8"))
        writer.write(body_bytes)
        await writer.drain()

    async def start(self) -> None:
        """Start the gateway server."""
        host = self.config.host
        port = self.config.port
        logger.info(f"Starting LLM Gateway on {host}:{port}")

        server = await asyncio.start_server(
            self.handle_request,
            host=host,
            port=port,
        )
        self._server = server

        addr = server.sockets[0].getsockname()
        logger.info(f"LLM Gateway listening on {addr}")

        async with server:
            await server.serve_forever()

    async def stop(self) -> None:
        """Stop the gateway server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()


def main() -> None:
    """Main entry point."""
    config_path = os.environ.get("GATEWAY_CONFIG")
    config = GatewayConfig.load(config_path)

    # Update log level
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    server = GatewayServer(config)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(server.stop())


if __name__ == "__main__":
    main()
