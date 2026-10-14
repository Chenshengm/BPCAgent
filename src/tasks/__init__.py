"""Application tasks, separated from agent execution strategies."""

from .base import Task, TaskResult
from .patient_state import PatientStateTask, PatientStateTaskConfig

__all__ = ["Task", "TaskResult", "PatientStateTask", "PatientStateTaskConfig"]
