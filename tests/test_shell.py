"""Exercise real local processes and the BasicAgent loop without a model service."""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from bpcagent import (
    BasicAgent, Config, LLMResponse, ShellTool, ShellToolConfig, ToolCall, ToolRegistry,
)


@unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "POSIX shell required")
class ShellToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="bpc shell 中文 ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.python = shlex.quote(sys.executable)
        self.registry = ToolRegistry()
        self.tool = ShellTool(ShellToolConfig(
            work_dir=self.directory, termination_grace_seconds=0.1,
        ))
        self.registry.register_tool(self.tool)

    def execute(self, command):
        result = self.registry.execute_tool("run_shell", {"command": command})
        self.assertEqual(result.status, "success", result.error)
        return result.data

    def configure(self, **options):
        self.tool = ShellTool(ShellToolConfig(
            work_dir=self.directory, termination_grace_seconds=0.1, **options,
        ))
        self.registry.register_tool(self.tool, replace=True)

    def test_multiline_read_write_modify_and_redirection(self):
        data = self.execute("cat > '结果 文件.txt' <<'END'\n第一行\nEND\nprintf '第二行\\n' >> '结果 文件.txt'\ncat '结果 文件.txt' | cat")
        self.assertEqual(data["stdout"], "第一行\n第二行\n")
        self.assertEqual((self.directory / "结果 文件.txt").read_text(), data["stdout"])
        self.assertEqual(data["cwd"], str(self.directory))
        self.assertEqual(data["exit_code"], 0)

    def test_new_process_retains_files_but_not_cwd_or_variables(self):
        self.execute("mkdir child\ncd child\nexport BPC_TRANSIENT=value\nprintf saved > file.txt")
        data = self.execute("printf '%s\\n' \"${BPC_TRANSIENT-unset}\"\npwd\ncat child/file.txt")
        self.assertEqual(data["stdout"], f"unset\n{self.directory}\nsaved")

    def test_nonzero_exit_is_command_result_not_tool_error(self):
        data = self.execute("printf partial; printf problem >&2; exit 7")
        self.assertEqual((data["exit_code"], data["stdout"], data["stderr"]), (7, "partial", "problem"))
        self.assertFalse(data["timed_out"])

    def test_stdin_is_closed(self):
        self.assertEqual(self.execute("cat")["stdout"], "")

    def test_invalid_arguments_do_not_launch_process(self):
        with patch("bpcagent.tools.shell.subprocess.Popen") as launch:
            for args in ({}, {"command": " \n"}, {"command": 3}, {"command": "a\x00b"},
                         {"command": "pwd", "cwd": "/"}):
                with self.subTest(args=args):
                    result = self.registry.execute_tool("run_shell", args)
                    self.assertEqual(result.error.code, "invalid_arguments")
            launch.assert_not_called()

    def test_start_failure_is_execution_error(self):
        with patch("bpcagent.tools.shell.subprocess.Popen", side_effect=OSError("cannot launch")):
            result = self.registry.execute_tool("run_shell", {"command": "pwd"})
        self.assertEqual(result.error.code, "execution_error")

    def test_configuration_validation(self):
        for extra in ({"timeout_seconds": True}, {"timeout_seconds": 0},
                      {"timeout_seconds": float("inf")}, {"max_output_bytes": False},
                      {"env_overrides": {"BASH_ENV": "startup.sh"}},
                      {"env_overrides": {"BAD=NAME": "value"}}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                ShellToolConfig(work_dir=self.directory, **extra)
        with self.assertRaises(FileNotFoundError):
            ShellTool(ShellToolConfig(work_dir=self.directory / "absent"))

    def test_environment_does_not_inherit_credentials_or_startup_hooks(self):
        hook = self.directory / "startup.sh"
        hook.write_text("printf ran > startup-ran")
        with patch.dict(os.environ, {"LLM_API_KEY": "fake-secret", "BASH_ENV": str(hook)}):
            self.configure(env_overrides={"BPC_EXPLICIT": "yes"})
            data = self.execute("printf '%s:%s' \"${LLM_API_KEY-unset}\" \"$BPC_EXPLICIT\"")
        self.assertEqual(data["stdout"], "unset:yes")
        self.assertFalse((self.directory / "startup-ran").exists())

    def test_large_both_streams_are_drained_and_capped(self):
        self.configure(max_output_bytes=1024)
        data = self.execute(f"{self.python} - <<'PY'\nimport os\nfor _ in range(100):\n os.write(1, b'x' * 8192)\n os.write(2, b'y' * 8192)\nPY")
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["stdout"], "x" * 1024)
        self.assertEqual(data["stderr"], "y" * 1024)
        self.assertTrue(data["stdout_truncated"] and data["stderr_truncated"])

    def test_invalid_utf8_remains_json_serializable(self):
        data = self.execute(f"{self.python} -c 'import os; os.write(1, bytes([255]))'")
        self.assertEqual(data["stdout"], "\ufffd")
        json.dumps(data, allow_nan=False)

    def test_timeout_kills_stubborn_child_and_preserves_partial_output(self):
        self.configure(timeout_seconds=0.25)
        command = f"{self.python} - <<'PY'\nimport signal, time\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('started', flush=True)\ntime.sleep(0.8)\nPath('late-write').write_text('bad')\nPY"
        start = time.monotonic()
        data = self.execute(command)
        self.assertTrue(data["timed_out"])
        self.assertIn("started", data["stdout"])
        self.assertNotEqual(data["exit_code"], 0)
        self.assertLess(time.monotonic() - start, 3)
        time.sleep(0.9)
        self.assertFalse((self.directory / "late-write").exists())

    def test_normal_exit_cleans_background_children_holding_pipes(self):
        data = self.execute("(sleep 0.8; printf bad > late-background) &\nprintf done")
        self.assertEqual(data["stdout"], "done")
        self.assertEqual(data["exit_code"], 0)
        self.assertFalse(data["timed_out"])
        time.sleep(0.9)
        self.assertFalse((self.directory / "late-background").exists())

    def test_collector_failure_still_reaps_process(self):
        launched = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            launched.append(process)
            return process

        with patch("bpcagent.tools.shell.subprocess.Popen", side_effect=launch), \
             patch.object(self.tool, "_collect_output", side_effect=OSError("read failed")):
            result = self.registry.execute_tool("run_shell", {"command": "sleep 30"})
        self.assertEqual(result.error.code, "execution_error")
        self.assertEqual(result.error.message, "read failed")
        self.assertIsNotNone(launched[0].poll())
        self.assertTrue(launched[0].stdout.closed and launched[0].stderr.closed)

    def test_denied_existence_probe_does_not_skip_forceful_cleanup(self):
        self.configure(timeout_seconds=0.25)
        real_killpg = os.killpg
        sent = []

        def killpg(pid, sig):
            sent.append(sig)
            if sig == 0:
                raise PermissionError("probe denied")
            return real_killpg(pid, sig)

        with patch("bpcagent.tools.shell.os.killpg", side_effect=killpg):
            data = self.execute("trap '' TERM\nwhile :; do :; done")
        self.assertTrue(data["timed_out"])
        self.assertIn(signal.SIGKILL, sent)
        self.assertEqual(data["exit_code"], -signal.SIGKILL)

    def test_basic_agent_receives_results_and_writes_real_file(self):
        llm = Mock()
        llm.invoke.side_effect = [
            LLMResponse(tool_calls=[ToolCall(
                id=f"shell_{index}", name="run_shell", arguments=json.dumps({"command": command}),
            )], finish_reason="tool_calls")
            for index, command in enumerate(("printf 'hello' > result.txt", "cat result.txt"))
        ] + [LLMResponse(content="Verified result.txt", finish_reason="stop")]
        agent = BasicAgent("ShellExperimentAgent", llm=llm, tool_registry=self.registry,
                           config=Config.resolve(environ={}, overrides={"max_tool_iterations": 4}))
        with patch.object(agent, "_save_history"):
            result = agent.run("Write a file and read it back")
        self.assertTrue(result.finished)
        self.assertEqual((self.directory / "result.txt").read_text(), "hello")
        tool_messages = [item for item in result.transcript if item["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(json.loads(tool_messages[-1]["content"])["data"]["stdout"], "hello")


if __name__ == "__main__":
    unittest.main()
