"""一次 Agent 运行产生的内容与交互记录。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """保存运行产物；任务评价与奖励由 Task 层的 TaskResult 负责。

    finished 为 True/False 时表示已知的运行完成状态；None 表示未跟踪。
    未跟踪完成状态不等于失败，也不自动决定该次运行能否用于训练。
    """

    output: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    finished: bool | None = None
