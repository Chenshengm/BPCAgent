"""Application-owned sandbox settings; never supplied by a tool call."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    image: str = Field(default="bpcagent-sandbox:0.1", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/:@-]*$")
    docker_path: Path | None = None
    command_timeout: float = Field(default=60.0, gt=0, allow_inf_nan=False, strict=True)
    control_timeout: float = Field(default=15.0, gt=0, allow_inf_nan=False, strict=True)
    memory_mb: int = Field(default=512, ge=64, strict=True)
    cpus: float = Field(default=1.0, gt=0, allow_inf_nan=False, strict=True)
    pids_limit: int = Field(default=64, ge=8, strict=True)
    max_output_bytes: int = Field(default=32768, gt=0, strict=True)
    max_workspace_bytes: int = Field(default=256 * 1024 * 1024, gt=0, strict=True)
    max_workspace_files: int = Field(default=10000, gt=0, strict=True)
    max_artifact_bytes: int = Field(default=16 * 1024 * 1024, gt=0, strict=True)
