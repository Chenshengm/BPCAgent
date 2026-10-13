"""The same run_shell action, executed only inside a Docker sandbox."""

from .tool import Tool
from .shell import ShellArgs
from ..sandbox import DockerSandbox


class SandboxShellTool(Tool[ShellArgs]):
    args_schema = ShellArgs

    def __init__(self, sandbox: DockerSandbox):
        super().__init__("run_shell", (
            "Execute a non-interactive shell command in an isolated container at /workspace. "
            "Read input/; write only scripts/, output/ and tmp/. Network access is disabled. "
            "Each call uses a fresh container and shell; only workspace files persist. "
            "Use /usr/local/bin/python. Do not start background services or install packages. "
            "Check exit_code, timed_out and truncated output, then verify the result files."
        ))
        self.sandbox = sandbox

    def run(self, parameters: ShellArgs):
        return self.sandbox.execute(parameters.command)
