from inspect import Parameter, signature
from typing import Any, Callable, get_type_hints

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.mcp.handlers.tool_handler import ToolHandler
import asyncio


def _generate_input_schema(func: Callable) -> dict[str, Any]:
    """从函数签名生成 JSON Schema"""
    try:
        sig = signature(func)
        hints = get_type_hints(func)
    except (ValueError, TypeError):
        return {"type": "object", "properties": {}}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Get type annotation
        param_type = hints.get(param_name, param.annotation)
        json_type = _python_type_to_json_type(param_type)

        prop: dict[str, Any] = {"type": json_type}

        # Handle description from docstring if available
        if func.__doc__:
            doc = func.__doc__
            if param_name in doc:
                # Simple extraction - look for param name in docstring
                pass

        # Handle default values
        if param.default is not Parameter.empty:
            default_val = param.default
            if default_val is None:
                prop["default"] = None
            elif isinstance(default_val, bool):
                prop["default"] = default_val
            elif isinstance(default_val, (int, float, str)):
                prop["default"] = default_val
            elif isinstance(default_val, (list, dict)):
                prop["default"] = default_val
        else:
            required.append(param_name)

        properties[param_name] = prop

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required

    return result


def _python_type_to_json_type(py_type: Any) -> str:
    """将 Python 类型映射到 JSON Schema 类型"""
    if py_type is Any:
        return "string"

    # Handle Optional[X] -> X
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        from typing import Union
        if origin is Union:
            args = getattr(py_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _python_type_to_json_type(non_none[0])

    # Handle list[X] -> array
    if origin is list:
        return "array"

    # Handle dict[K, V] -> object
    if origin is dict:
        return "object"

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    if py_type in type_map:
        return type_map[py_type]

    # Handle generic types like List[str], Dict[str, Any]
    if hasattr(py_type, "__origin__"):
        return _python_type_to_json_type(py_type.__origin__)

    return "string"


class NovelKBMCPServer:
    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self.config = config
        self.tool_handler = ToolHandler(config)
        self.app = Server("novel-kb")
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP tool handlers"""
        
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=_generate_input_schema(tool.func),
                )
                for tool in self.tool_handler.tools
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict):
            """Call a tool"""
            import json
            tool = next((t for t in self.tool_handler.tools if t.name == name), None)
            if not tool:
                raise ValueError(f"Tool not found: {name}")
            
            # 处理异步和同步函数
            result = tool.func(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            
            # 转换为JSON格式的文本
            result_text = json.dumps(result, ensure_ascii=False, indent=2)
            
            return [TextContent(type="text", text=result_text)]

    def run(self) -> None:
        async def _start() -> None:
            async with stdio_server() as (read_stream, write_stream):
                init_opts = self.app.create_initialization_options()
                await self.app.run(read_stream, write_stream, init_opts)

        asyncio.run(_start())

