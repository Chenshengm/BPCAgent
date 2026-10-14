"""Small task interface and result container; no execution policy in the base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.result import AgentResult


@dataclass
class TaskResult:
    """Task status and existing artifacts, with the original agent result retained.

    Files may exist even when finished is False. Their presence alone does not
    establish success. Task-level diagnostics live in info; agent_result keeps
    the model report, transcript and original execution status unchanged.
    """

    finished: bool = False
    task_dir: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    log_path: Path | None = None
    agent_result: AgentResult | None = None
    info: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """One run is one task. Concrete tasks define their own query arguments.

    Return known execution failures in TaskResult. Configuration and unexpected
    programming errors may propagate. Printing and process exit belong to the
    application, not this interface.
    """

    @abstractmethod
    def run(self, **kwargs) -> TaskResult:
        """Execute an independent task and return structured status and artifacts."""
        raise NotImplementedError
