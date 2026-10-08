# --- 异常系统 ---

class BasicException(Exception):
    """基础异常类"""
    pass

class LLMException(BasicException):
    """LLM相关异常"""
    pass

class AgentException(BasicException):
    """Agent 异常；code 标识可转换为运行结果的已知失败。"""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code

class ConfigException(BasicException):
    """配置相关异常"""
    pass

class ToolException(BasicException):
    """工具相关异常"""
    pass
