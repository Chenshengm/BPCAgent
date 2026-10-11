"""工具参数、OpenAI 函数工具定义以及统一注册和执行入口。"""

from abc import ABC, abstractmethod
import inspect
import json
import math
import re
from typing import Any, Callable, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..agents.exceptions import ToolException


class ToolArgs(BaseModel):
    """工具参数模型；默认严格校验类型、拒绝未知字段，并校验默认值。"""

    model_config = ConfigDict(
        extra="forbid", strict=True, validate_default=True,
        allow_inf_nan=False, hide_input_in_errors=True,
    )


ArgsT = TypeVar("ArgsT", bound=ToolArgs)
_ARGUMENT_LOG_MAX_CHARS = 200


def _json_data(value: Any) -> Any:
    """复制原生 JSON 数据；不自动把任意对象、元组或非字符串键转成文本。"""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json_data(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_data(item) for key, item in value.items()}
    raise ValueError("Tool results must be native JSON data with finite numbers and string dictionary keys")


class ToolError(BaseModel):
    """供程序判断的错误类别及简短说明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: Literal["tool_not_found", "invalid_arguments", "execution_error", "serialization_error"]
    message: str


class ToolResult(BaseModel):
    """一次工具执行的结果；是否成功由 status 判断，成功数据可以为 None。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str = Field(default_factory=lambda: f"call_{uuid4().hex}", min_length=1)
    tool_name: str
    status: Literal["success", "error"]
    data: Any = None
    error: ToolError | None = None

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value):
        """结果必须能无损表示为 JSON 数据。"""
        return _json_data(value)

    @model_validator(mode="after")
    def validate_status(self):
        """禁止成功携带错误或失败携带成功数据。"""
        if self.status == "success" and self.error is not None:
            raise ValueError("Successful results must not contain an error")
        if self.status == "error" and (self.error is None or self.data is not None):
            raise ValueError("Failed results must include an error and data must be None")
        return self

    @classmethod
    def failure(cls, tool_name: str, code: str, message: str, *, call_id: str | None = None):
        """统一创建失败结果，保留已分配的调用 ID。"""
        return cls(
            tool_name=tool_name, status="error", error=ToolError(code=code, message=message[:500]),
            **({"call_id": call_id} if call_id is not None else {}),
        )


class Tool(ABC, Generic[ArgsT]):
    """工具定义；run 接收校验后的参数模型，框架通过 ToolRegistry 执行。"""

    args_schema: type[ArgsT]

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: ArgsT) -> Any:
        """返回原生 JSON 数据；执行失败应抛出异常，不能伪装成成功文本。"""
        raise NotImplementedError

    def to_openai_tool(self) -> dict[str, Any]:
        """导出 Chat Completions 函数工具定义；本地严格校验与服务端 strict 分开。

        当前使用 strict=False，保留参数默认值与可省略字段的本地语义。
        此方法不调用服务，也不承诺任意 Schema 都满足服务端严格模式子集。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
                "strict": False,
            },
        }


class FunctionTool(Tool[ArgsT]):
    """将同步函数包装成普通工具，按参数字段名传入关键字参数。"""

    def __init__(self, name: str, description: str, func: Callable[..., Any], *, args_schema: type[ArgsT]):
        super().__init__(name, description)
        if not isinstance(args_schema, type) or not issubclass(args_schema, ToolArgs):
            raise ToolException("args_schema must be a subclass of ToolArgs")
        if not callable(func) or inspect.iscoroutinefunction(func) or inspect.isgeneratorfunction(func):
            raise ToolException("Function tools must be regular synchronous functions")
        try:
            signature = inspect.signature(func)
            if any(param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL, param.VAR_KEYWORD)
                   for param in signature.parameters.values()):
                raise ValueError("Function parameters must be explicitly named")
            signature.bind(**dict.fromkeys(args_schema.model_fields))
        except (TypeError, ValueError) as exc:
            raise ToolException("The function signature must accept every argument model field and require no additional parameters") from exc
        self.func = func
        self.args_schema = args_schema

    def run(self, parameters: ArgsT) -> Any:
        """保留嵌套模型的类型，避免 model_dump 将其提前转为字典。"""
        return self.func(**{name: getattr(parameters, name) for name in self.args_schema.model_fields})


class ToolRegistry:
    """用同一个字典管理对象工具和函数工具，集中校验、执行和包装结果。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool, *, replace: bool = False) -> None:
        """注册工具；名称遵循 OpenAI 函数名称规则，重名必须显式替换。"""
        if not isinstance(tool, Tool):
            raise ToolException("Only Tool instances can be registered")
        if not isinstance(tool.name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tool.name):
            raise ToolException("Tool names must contain 1 to 64 letters, digits, underscores, or hyphens")
        schema = getattr(tool, "args_schema", None)
        if not isinstance(schema, type) or not issubclass(schema, ToolArgs):
            raise ToolException("Tools must declare a ToolArgs subclass as args_schema")
        if inspect.iscoroutinefunction(tool.run) or inspect.isgeneratorfunction(tool.run):
            raise ToolException("The tool interface supports synchronous execution only")
        try:
            tool.to_openai_tool()
        except Exception as exc:
            raise ToolException("The tool argument model could not generate a JSON Schema") from exc
        if tool.name in self._tools and not replace:
            raise ToolException(f"Tool '{tool.name}' is already registered; use replace=True to replace it")
        self._tools[tool.name] = tool
        print(f"✅ Tool '{tool.name}' registered.")

    def register_function(self, name: str, description: str, func: Callable[..., Any], *,
                          args_schema: type[ToolArgs], replace: bool = False) -> None:
        """包装函数后走同一注册入口，不另行存储函数。"""
        self.register_tool(FunctionTool(name, description, func, args_schema=args_schema), replace=replace)

    def get_tool(self, name: str) -> Tool | None:
        """按名称查找已注册工具。"""
        return self._tools.get(name)

    def unregister_tool(self, name: str) -> None:
        """删除工具；名称不存在时不执行额外操作。"""
        if self._tools.pop(name, None) is not None:
            print(f"🗑️ Tool '{name}' unregistered.")
        else:
            print(f"⚠️ Tool '{name}' does not exist.")

    def list_tools(self) -> list[str]:
        """按注册顺序列出名称。"""
        return list(self._tools)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """返回可用于 Chat Completions tools 参数的函数定义列表。"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def get_descriptions(self) -> str:
        """返回工具定义的 JSON 文本，供展示和检查参数 Schema。"""
        return json.dumps(self.to_openai_tools(), ensure_ascii=False) if self._tools else "No tools available"

    def execute_tool(self, name: str, arguments: dict[str, Any], *, call_id: str | None = None) -> ToolResult:
        """校验后执行一次；失败不自动重试，结果序列化失败也不会重新执行工具。"""
        identity = call_id if call_id is not None else f"call_{uuid4().hex}"
        if not isinstance(identity, str) or not identity:
            raise ToolException("call_id must be a nonempty string")
        tool = self.get_tool(name)
        if tool is None:
            print(f"❌ Tool '{name}' is not registered.")
            return ToolResult.failure(name, "tool_not_found", f"Tool '{name}' is not registered", call_id=identity)
        if not isinstance(arguments, dict):
            print(f"❌ Tool '{name}' requires JSON object arguments; execution skipped.")
            return ToolResult.failure(name, "invalid_arguments", "Tool arguments must be a JSON object", call_id=identity)
        try:
            parameters = tool.args_schema.model_validate(_json_data(arguments), strict=True)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'arguments'}: {error['msg']}"
                for error in exc.errors(include_input=False, include_url=False, include_context=False)
            )
            print(f"❌ Tool '{name}' argument validation failed; execution skipped: {details}")
            return ToolResult.failure(name, "invalid_arguments", details, call_id=identity)
        except Exception:
            print(f"❌ Tool '{name}' argument validation failed; execution skipped.")
            return ToolResult.failure(name, "invalid_arguments", "Argument validation failed", call_id=identity)
        try:
            argument_preview = str(parameters)
            if len(argument_preview) > _ARGUMENT_LOG_MAX_CHARS:
                total_chars = len(argument_preview)
                argument_preview = argument_preview[:_ARGUMENT_LOG_MAX_CHARS] + f"... [truncated; {total_chars} chars total]"
            print(f"🔧 Executing tool '{name}' with arguments: {argument_preview}")
            data = tool.run(parameters)
            if inspect.isawaitable(data):
                if inspect.iscoroutine(data):
                    data.close()
                raise ToolException("The synchronous interface cannot execute asynchronous tools")
        except Exception as exc:
            print(f"❌ Tool '{name}' execution failed: {exc}")
            return ToolResult.failure(name, "execution_error", str(exc), call_id=identity)
        try:
            result = ToolResult(call_id=identity, tool_name=name, status="success", data=data)
        except (ValidationError, ValueError, TypeError, RecursionError):
            print(f"❌ Tool '{name}' executed, but its return value cannot be serialized.")
            return ToolResult.failure(name, "serialization_error", "The tool executed, but its return value is not serializable native JSON data", call_id=identity)
        return result
