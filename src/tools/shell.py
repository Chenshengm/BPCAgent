"""Synchronous local shell execution for macOS/Linux; this is not a sandbox."""

from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .tool import Tool, ToolArgs


class ShellArgs(ToolArgs):
    command: str = Field(description="Non-interactive shell command; multiline scripts are supported.")

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Command must be nonblank and contain no NUL bytes")
        return value  # Preserve indentation and quoting, including heredocs.


class ShellToolConfig(BaseModel):
    """Application-owned settings, independent of the LLM request configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True, hide_input_in_errors=True)

    work_dir: Path
    shell_path: Path = Path("/bin/bash")
    timeout_seconds: float = Field(30.0, gt=0, allow_inf_nan=False)
    termination_grace_seconds: float = Field(2.0, gt=0, allow_inf_nan=False)
    max_output_bytes: int = Field(32768, gt=0, strict=True)
    env_overrides: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("timeout_seconds", "termination_grace_seconds", mode="before")
    @classmethod
    def validate_seconds(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Timeouts must be numeric, not booleans or strings")
        return value

    @field_validator("env_overrides")
    @classmethod
    def validate_environment_variables(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if not key or "=" in key or "\x00" in key or "\x00" in item:
                raise ValueError("Invalid environment variable name or value")
            if key in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}:
                raise ValueError("Shell startup hooks cannot be configured")
        return value


class ShellTool(Tool[ShellArgs]):
    """Each call owns one process group. Files persist, shell state does not.

    Commands run with the host user's permissions. work_dir is a starting
    directory, NOT a filesystem or network security boundary. Writes are not
    rolled back on failure. Detached processes require OS-level isolation.
    """

    args_schema = ShellArgs

    def __init__(self, config: ShellToolConfig):
        super().__init__(
            name="run_shell",
            description=(
                "Execute a non-interactive command in the configured local working directory. "
                "Read and write files, run scripts, and use pipes or redirection. Each call starts "
                "a fresh shell: files persist, but cd and exported variables do not. "
                "Do not launch background services or interactive programs. Check timed_out "
                "and exit_code, then verify written files before reporting completion. "
                "Output is capped; redirect large results to files. Failed commands may have written files."
            ),
        )
        self.config = config.model_copy(deep=True)
        self._work_dir, self._shell_path = self._validate_environment()
        self._environment = self._build_environment()

    def _validate_environment(self) -> tuple[Path, Path]:
        if sys.platform != "darwin" and not sys.platform.startswith("linux"):
            raise RuntimeError("ShellTool supports macOS and Linux only")
        work_dir = self.config.work_dir.expanduser().resolve(strict=True)
        if not work_dir.is_dir():
            raise NotADirectoryError(str(work_dir))
        shell_path = self.config.shell_path.expanduser().resolve(strict=True)
        if not shell_path.is_file() or not os.access(shell_path, os.X_OK):
            raise ValueError("shell_path must point to an executable shell")
        return work_dir, shell_path

    def _build_environment(self) -> dict[str, str]:
        # Never inherit os.environ wholesale (it commonly holds LLM credentials).
        environment = {
            "PATH": os.defpath,
            "LANG": "en_US.UTF-8" if sys.platform == "darwin" else "C.UTF-8",
            "TMPDIR": str(self._work_dir),
            "PAGER": "cat",
            "TERM": "dumb",
            "NO_COLOR": "1",
        }
        environment.update(self.config.env_overrides)
        return environment

    def run(self, parameters: ShellArgs) -> dict[str, Any]:
        return self._execute_command(parameters.command)

    def _execute_command(self, command: str) -> dict[str, Any]:
        start = time.monotonic()
        argv = [str(self._shell_path)]
        if self._shell_path.name == "bash":
            argv += ["--noprofile", "--norc"]
        argv += ["-c", command]
        process = subprocess.Popen(
            argv, cwd=self._work_dir, env=self._environment.copy(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr, truncated, timed_out = self._collect_output(process, start)
        finally:
            try:
                self._terminate_process_group(process)
            finally:
                process.stdout.close()
                process.stderr.close()
        return {
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "timed_out": timed_out,
            "duration_ms": round((time.monotonic() - start) * 1000),
            "stdout_truncated": truncated[0],
            "stderr_truncated": truncated[1],
            "cwd": str(self._work_dir),
        }

    def _collect_output(
        self, process: subprocess.Popen, start: float,
    ) -> tuple[bytearray, bytearray, list[bool], bool]:
        buffers = [bytearray(), bytearray()]
        truncated = [False, False]
        timed_out = False
        drain_deadline = None
        with selectors.DefaultSelector() as selector:
            for index, stream in enumerate((process.stdout, process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map() or process.poll() is None:
                if drain_deadline is None:
                    if process.poll() is not None:
                        # A background child can keep pipes open after bash exits.
                        self._terminate_process_group(process)
                        drain_deadline = time.monotonic() + 0.5
                    elif time.monotonic() - start >= self.config.timeout_seconds:
                        timed_out = True
                        self._terminate_process_group(process)
                        drain_deadline = time.monotonic() + 0.5
                elif time.monotonic() >= drain_deadline:
                    # Even an escaped child holding a pipe must not hang the caller.
                    for key in selector.get_map().values():
                        truncated[key.data] = True
                    break
                for key, _ in selector.select(timeout=0.05):
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    index = key.data
                    remaining = self.config.max_output_bytes - len(buffers[index])
                    buffers[index].extend(chunk[:remaining])
                    truncated[index] |= len(chunk) > remaining
        return buffers[0], buffers[1], truncated, timed_out

    def _terminate_process_group(self, process: subprocess.Popen) -> None:
        """Stop ordinary descendants as well as the shell; always reap the shell."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + self.config.termination_grace_seconds
        while time.monotonic() < deadline:
            process.poll()  # Reap an exited leader before probing its group.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                # Signal 0 is only an existence probe. EPERM does not mean the
                # group disappeared; still attempt SIGKILL after the grace period.
                pass
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)
