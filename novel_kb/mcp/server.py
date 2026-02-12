from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.mcp.handlers.tool_handler import ToolHandler


class NovelKBMCPServer:
    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self.config = config
        self.tool_handler = ToolHandler(config)

    def run(self) -> None:
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
        except ImportError as exc:
            raise RuntimeError("Missing dependency: mcp") from exc

        server = Server("novel-kb")
        self.tool_handler.register(server)
        stdio_server(server).run()
