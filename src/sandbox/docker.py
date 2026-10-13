"""One local, restricted Docker container per command; only task files persist."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import shutil
import stat
import subprocess
import time
from uuid import uuid4

from .config import SandboxConfig


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    directories = ("input", "scripts", "output", "tmp")

    def __init__(self, config: SandboxConfig, task_dir: Path | str):
        self.config = config
        self.task_dir = Path(task_dir).expanduser().resolve(strict=True)
        self.session_id = uuid4().hex
        self.image_id = None
        self.context = None
        self.endpoint = None
        self.executions: list[dict] = []
        self.broken = False
        self._active: set[str] = set()
        self._docker = None
        # Docker needs its user configuration, but not LLM keys or proxy variables.
        self._environment = {"HOME": str(Path.home()), "PATH": os.defpath}
        self._validate_directories()

    def _validate_directories(self):
        if "," in str(self.task_dir) or "\n" in str(self.task_dir):
            raise SandboxError("Task paths must not contain commas or newlines (Docker mount syntax)")
        for name in self.directories:
            directory = self.task_dir / name
            if directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory:
                raise SandboxError(f"Expected a real task directory: {directory}")

    def _control(self, args, *, check=True, use_context=True):
        prefix = [self._docker]
        if use_context and self.context:
            prefix += ["--context", self.context]
        try:
            result = subprocess.run(
                prefix + list(args), shell=False, env=self._environment,
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=self.config.control_timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"Docker control operation failed: {exc}") from exc
        if check and result.returncode:
            raise SandboxError(f"Docker {args[0]} failed: {result.stderr.strip()[:1000]}")
        return result

    def check_available(self):
        if self.broken:
            raise SandboxError("Sandbox is unusable after an earlier infrastructure/cleanup failure")
        candidates = [self.config.docker_path] if self.config.docker_path else [
            shutil.which("docker"), Path.home() / ".docker/bin/docker",
            Path("/Applications/Docker.app/Contents/Resources/bin/docker"),
        ]
        self._docker = next((str(Path(p).resolve()) for p in candidates
                             if p and Path(p).is_file() and os.access(p, os.X_OK)), None)
        if not self._docker:
            raise SandboxError("Docker CLI not found; install/start Docker Desktop or set docker_path")
        self.context = self._control(["context", "show"], use_context=False).stdout.strip()
        context = json.loads(self._control(["context", "inspect", self.context], use_context=False).stdout)[0]
        self.endpoint = context["Endpoints"]["docker"]["Host"]
        if not self.endpoint.startswith("unix://"):
            raise SandboxError("Only a local Unix-socket Docker context is allowed; remote Docker is unsupported")
        info = json.loads(self._control(["info", "--format", "{{json .}}"]).stdout)
        if info["OSType"] != "linux":
            raise SandboxError("A local Linux Docker engine is required")
        image = json.loads(self._control(["image", "inspect", self.config.image]).stdout)[0]
        self.image_id = image["Id"]  # Pin this run to actual content, even if the tag later changes.
        self._validate_directories()
        self._check_workspace()
        if os.getuid() == 0:
            raise SandboxError("Run BPCAgent as a non-root host user for bind-mount ownership")
        return self.metadata()

    def metadata(self):
        return {
            "backend": "docker", "lifecycle": "container_per_command",
            "session_id": self.session_id, "context": self.context, "endpoint": self.endpoint,
            "image": self.config.image, "image_id": self.image_id,
            "limits": self.config.model_dump(mode="json", exclude={"docker_path", "image"}),
            "network": "none", "read_only_rootfs": True,
            "broken": self.broken, "active_containers": sorted(self._active),
            "executions": list(self.executions),
        }

    def _create_args(self, name, command):
        args = [
            "create", "--pull", "never", "--name", name,
            "--label", f"bpcagent.sandbox.session={self.session_id}",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--init",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--memory", f"{self.config.memory_mb}m", "--memory-swap", f"{self.config.memory_mb}m",
            "--cpus", str(self.config.cpus), "--pids-limit", str(self.config.pids_limit),
            "--ulimit", "core=0", "--log-driver", "none", "--workdir", "/workspace",
            "--env", "HOME=/workspace/tmp", "--env", "TMPDIR=/workspace/tmp",
            "--env", "TMP=/workspace/tmp", "--env", "TEMP=/workspace/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONUNBUFFERED=1",
            "--env", "LANG=C.UTF-8", "--env", "BASH_ENV=/dev/null", "--env", "ENV=/dev/null",
        ]
        for directory in self.directories:
            mount = f"type=bind,src={self.task_dir / directory},dst=/workspace/{directory}"
            if directory == "input":
                mount += ",readonly"
            args += ["--mount", mount]
        return args + ["--entrypoint", "/bin/bash", self.image_id, "--noprofile", "--norc", "-c", command]

    def _check_workspace(self):
        """Soft disk guard; no links followed. This is not an atomic disk quota."""
        total = count = 0
        self._validate_directories()

        def visit(fd):
            nonlocal total, count
            with os.scandir(fd) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:  # A running command may remove a temporary file.
                        continue
                    count += 1
                    if stat.S_ISDIR(info.st_mode):
                        # Open relative to the parent descriptor: a command cannot race
                        # a directory-to-symlink replacement into a host traversal.
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            visit(child)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                    else:
                        raise SandboxError("Workspace contains a symlink or special file; only regular files/directories are supported")
                    if total > self.config.max_workspace_bytes or count > self.config.max_workspace_files:
                        raise SandboxError("Workspace size/file-count limit exceeded (soft limit)")
        for name in self.directories:
            fd = os.open(self.task_dir / name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                visit(fd)
            finally:
                os.close(fd)

    def _collect_output(self, process):
        start = time.monotonic()
        buffers = [bytearray(), bytearray()]
        truncated = [False, False]
        timed_out = False
        last_check = start
        with selectors.DefaultSelector() as selector:
            for index, stream in enumerate((process.stdout, process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if now - start >= self.config.command_timeout:
                    timed_out = True
                    break
                if now - last_check >= 0.25:
                    self._check_workspace()
                    last_check = now
                for key, _ in selector.select(timeout=0.05):
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    i = key.data
                    remaining = self.config.max_output_bytes - len(buffers[i])
                    buffers[i].extend(chunk[:remaining])
                    truncated[i] |= len(chunk) > remaining
        return buffers, truncated, timed_out

    def _stop_and_remove(self, name):
        if name not in self._active:
            return
        result = self._control(["rm", "--force", name], check=False)
        if result.returncode and "No such container" not in result.stderr:
            self.broken = True
            raise SandboxError(f"Could not clean container {name}: {result.stderr[:500]}")
        self._active.discard(name)

    def execute(self, command: str):
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise ValueError("A nonblank shell command without NUL bytes is required")
        if self.broken:
            raise SandboxError("Sandbox is unusable; resolve the earlier failure before starting a new task")
        if self.image_id is None:
            self.check_available()
        self._check_workspace()
        name = f"bpcagent-sandbox-{self.session_id[:12]}-{uuid4().hex[:12]}"
        record = {"container_name": name, "image_id": self.image_id, "cleanup_status": "pending"}
        self.executions.append(record)
        self._active.add(name)  # Track before create: the CLI can fail after the daemon created it.
        started = time.monotonic()
        process = None
        result = None
        try:
            self._control(self._create_args(name, command))
            process = subprocess.Popen(
                [self._docker, "--context", self.context, "start", "--attach", name],
                shell=False, env=self._environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            buffers, truncated, timed_out = self._collect_output(process)
            state = {} if timed_out else json.loads(self._control([
                "inspect", "--format", "{{json .State}}", name,
            ]).stdout)
            if not timed_out and (state.get("Status") != "exited" or state.get("Running") or state.get("Error")):
                raise SandboxError(f"Container did not finish normally: {state}")
            result = {
                "exit_code": 124 if timed_out else state["ExitCode"],
                "stdout": buffers[0].decode("utf-8", errors="replace"),
                "stderr": buffers[1].decode("utf-8", errors="replace"),
                "timed_out": timed_out, "oom_killed": state.get("OOMKilled", False),
                "stdout_truncated": truncated[0], "stderr_truncated": truncated[1],
                "cwd": "/workspace", "container_name": name, "image_id": self.image_id,
            }
            record.update({key: result[key] for key in ("exit_code", "timed_out", "oom_killed")})
        except BaseException as exc:
            record["error"] = str(exc)[:1000]
            if not isinstance(exc, KeyboardInterrupt):
                self.broken = True
            raise
        finally:
            try:
                self._stop_and_remove(name)
                record["cleanup_status"] = "removed"
            except Exception as exc:
                self.broken = True
                record.update(cleanup_status="failed", cleanup_error=str(exc)[:1000])
                raise
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    process.stdout.close()
                    process.stderr.close()
                record["duration_ms"] = round((time.monotonic() - started) * 1000)
        try:
            self._check_workspace()
        except (OSError, SandboxError) as exc:
            self.broken = True
            record["error"] = str(exc)[:1000]
            raise SandboxError(str(exc)) from exc
        result.update(duration_ms=record["duration_ms"], cleanup_status=record["cleanup_status"])
        return result

    def read_artifact(self, directory: str, filename: str) -> bytes:
        """Read a bounded regular file only after all command containers are removed."""
        if self._active:
            raise SandboxError("Cannot read artifacts before container cleanup completes")
        if directory not in self.directories or Path(filename).name != filename:
            raise SandboxError("Invalid artifact path")
        self._validate_directories()
        path = self.task_dir / directory / filename
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SandboxError("Artifact must be a regular file, not a link or device")
            if info.st_size > self.config.max_artifact_bytes:
                raise SandboxError("Artifact is too large")
            data = handle.read(self.config.max_artifact_bytes + 1)
            if len(data) > self.config.max_artifact_bytes:
                raise SandboxError("Artifact is too large")
            return data

    def close(self):
        errors = []
        for name in list(self._active):
            try:
                self._stop_and_remove(name)
            except Exception as exc:
                self.broken = True
                errors.append(str(exc))
        if errors:
            raise SandboxError("; ".join(errors))

    def __enter__(self):
        self.check_available()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
