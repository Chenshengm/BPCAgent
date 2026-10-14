"""Patient extraction and sandbox normalization as a reusable application task."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import Task, TaskResult
from ..agents.config import Config
from ..agents.unit_patient_state import UnitPatientStateAgent
from ..sandbox import SandboxConfig


class PatientStateTaskConfig(BaseModel):
    """Configure once; relative paths are anchored when this config is created.

    Load dotenv in the application before creating PatientStateTask. The task
    snapshots model/environment settings on initialization and always enables
    sandbox execution. Paths are independent of the package installation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True,
                             hide_input_in_errors=True)

    data_dir: Path
    workspace_root: Path
    rules_path: Path | None = None
    agent_config: Config = Field(default_factory=lambda: Config(max_tool_iterations=10))
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    timezone: str = "Asia/Shanghai"

    @field_validator("data_dir", "workspace_root", "rules_path")
    @classmethod
    def resolve_path(cls, value):
        return value.expanduser().resolve() if value is not None else None

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value


class PatientStateTask(Task):
    """Extract state/evidence, normalize units in Docker, and collect artifacts.

    Each run creates a new UnitPatientStateAgent and a unique task directory.
    Unit rules, tool loops, container management and semantic validation remain
    owned by the existing agent/tool implementations.
    """

    def __init__(self, config: PatientStateTaskConfig):
        if not config.data_dir.is_dir():
            raise NotADirectoryError(f"Patient dataset directory does not exist: {config.data_dir}")
        if config.rules_path is not None and not config.rules_path.is_file():
            raise FileNotFoundError(f"Patient field rules do not exist: {config.rules_path}")
        if config.workspace_root.exists() and not config.workspace_root.is_dir():
            raise NotADirectoryError(f"Workspace root is not a directory: {config.workspace_root}")
        if any(c in str(config.workspace_root) for c in (",", "\n", "\r")):
            raise ValueError("Workspace root cannot contain commas or newlines (Docker mount syntax)")
        self.config = config.model_copy(update={"agent_config": Config.resolve(config.agent_config)})

    def run(self, *, patient_id: str, cutoff_day: int, cancer_id: str | None = None) -> TaskResult:
        self._validate_request(patient_id, cutoff_day, cancer_id)
        task_dir = self._create_work_dir(patient_id, cutoff_day)
        agent = self._create_agent(patient_id, cutoff_day, cancer_id, task_dir)
        agent_result = agent.run()
        artifacts = self._collect_artifacts(task_dir, cutoff_day)
        info = deepcopy(agent_result.info)
        finished = agent_result.finished is True
        required = {"state", "evidence", "normalized", "script"}
        if finished and (missing := required - artifacts.keys()):
            finished = False
            info.update(stop_reason="incomplete_artifacts",
                        error="Agent finished but required artifacts are missing: " + ", ".join(sorted(missing)))
        log_path = getattr(agent, "log_path", None)
        if log_path is not None:
            log_path = Path(log_path).resolve()
            if not log_path.is_file():
                log_path = None
        return TaskResult(finished=finished, task_dir=task_dir, artifacts=artifacts,
                          log_path=log_path, agent_result=agent_result, info=info)

    @staticmethod
    def _validate_request(patient_id, cutoff_day, cancer_id):
        if not isinstance(patient_id, str) or not patient_id.strip() or patient_id != patient_id.strip():
            raise ValueError("patient_id must be a nonblank string without surrounding whitespace")
        if patient_id in {".", ".."} or any(c in '/\\,:*?"<>|' or ord(c) < 32 or ord(c) == 127 for c in patient_id):
            raise ValueError("patient_id contains characters unsafe for a task directory name")
        # Negative days are valid in the existing extraction API; bool is not a day.
        if type(cutoff_day) is not int:
            raise ValueError("cutoff_day must be an integer day offset, not a date string or boolean")
        if cancer_id is not None and (not isinstance(cancer_id, str) or not cancer_id.strip()):
            raise ValueError("cancer_id must be a nonblank string when supplied")

    def _create_work_dir(self, patient_id, cutoff_day) -> Path:
        timestamp = datetime.now(ZoneInfo(self.config.timezone)).strftime("%Y%m%d_%H%M%S")
        directory = self.config.workspace_root / f"{timestamp}_{patient_id}_{cutoff_day}"
        # Do not reuse or overwrite another run, even with identical query arguments.
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def _create_agent(self, patient_id, cutoff_day, cancer_id, task_dir) -> UnitPatientStateAgent:
        return UnitPatientStateAgent(
            patient_id=patient_id, cutoff_day=cutoff_day, cancer_id=cancer_id,
            data_dir=self.config.data_dir, rules_path=self.config.rules_path,
            work_dir=task_dir, config=self.config.agent_config,
            sandbox_config=self.config.sandbox_config,
        )

    @staticmethod
    def _collect_artifacts(task_dir: Path, cutoff_day: int) -> dict[str, Path]:
        artifacts = {}

        def regular_file(path):
            return not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(task_dir)

        input_dir = task_dir / "input"
        if not input_dir.is_symlink() and input_dir.is_dir():
            for kind in ("state", "evidence"):
                pattern = re.compile(rf"patient_{kind}_day_{cutoff_day}_(\d+)\.json")
                candidates = []
                for path in input_dir.iterdir():
                    match = pattern.fullmatch(path.name)
                    if match and regular_file(path):
                        candidates.append((int(match[1]), path))
                if candidates:
                    artifacts[kind] = max(candidates, key=lambda item: item[0])[1]
        for key, directory, filename in (
            ("normalized", "output", "tumor_markers_standardized.json"),
            ("script", "scripts", "standardize_tm.py"),
        ):
            parent = task_dir / directory
            path = parent / filename
            if not parent.is_symlink() and regular_file(path):
                artifacts[key] = path
        return artifacts
