"""Offline sandbox checks: no Docker daemon and no model requests."""

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from bpcagent import DockerSandbox, SandboxConfig, SandboxError, SandboxShellTool, ToolRegistry


class SandboxTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for name in DockerSandbox.directories:
            (self.root / name).mkdir()
        self.sandbox = DockerSandbox(SandboxConfig(), self.root)

    def test_config_rejects_model_controlled_privilege_options(self):
        for values in ({"network_enabled": True}, {"mounts": []}, {"privileged": True},
                       {"command_timeout": 0}, {"cpus": True}, {"pids_limit": True},
                       {"image": "--privileged"}, {"max_output_bytes": 0}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                SandboxConfig(**values)

    def test_no_shell_interpolation_and_fixed_restrictions(self):
        command = "printf '%s' '$(touch /host-file)'; echo \"quoted\""
        self.sandbox.image_id = "sha256:fixed"
        args = self.sandbox._create_args("task-name", command)
        self.assertEqual(args[-1], command)
        self.assertEqual(args[-5:-1], ["sha256:fixed", "--noprofile", "--norc", "-c"])
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("--read-only", args)
        self.assertIn("no-new-privileges", args)
        self.assertIn("ALL", args)
        mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
        self.assertEqual(len(mounts), 4)
        self.assertTrue(mounts[0].endswith("/workspace/input,readonly"))
        self.assertNotIn("--privileged", args)

    def test_docker_cli_never_receives_model_secrets_or_host_shell(self):
        self.sandbox._docker = "/docker"
        with patch.dict(os.environ, {"LLM_API_KEY": "secret", "DOCKER_HOST": "tcp://remote:2375"}), \
                patch("bpcagent.sandbox.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "ok", "")
            self.sandbox._control(["info"])
        options = run.call_args.kwargs
        self.assertFalse(options["shell"])
        self.assertNotIn("LLM_API_KEY", options["env"])
        self.assertNotIn("DOCKER_HOST", options["env"])

    def test_remote_context_rejected_before_image_or_container_operations(self):
        self.sandbox.config = SandboxConfig(docker_path=Path("/bin/sh"))
        with patch.object(self.sandbox, "_control") as control:
            control.side_effect = [
                subprocess.CompletedProcess([], 0, "remote\n", ""),
                subprocess.CompletedProcess([], 0, json.dumps([{"Endpoints": {"docker": {"Host": "ssh://remote"}}}]), ""),
            ]
            with self.assertRaisesRegex(SandboxError, "local Unix"):
                self.sandbox.check_available()
            self.assertEqual(control.call_count, 2)

    def test_artifact_refuses_links_special_files_and_large_files(self):
        (self.root / "output" / "valid.json").write_bytes(b"{}")
        self.assertEqual(self.sandbox.read_artifact("output", "valid.json"), b"{}")
        (self.root / "output" / "link").symlink_to(self.root / "output" / "valid.json")
        with self.assertRaises(OSError):
            self.sandbox.read_artifact("output", "link")
        os.mkfifo(self.root / "output" / "fifo")
        with self.assertRaisesRegex(SandboxError, "regular file"):
            self.sandbox.read_artifact("output", "fifo")
        self.sandbox.config = SandboxConfig(max_artifact_bytes=1)
        with self.assertRaisesRegex(SandboxError, "too large"):
            self.sandbox.read_artifact("output", "valid.json")
        with self.assertRaises(SandboxError):
            self.sandbox.read_artifact("output", "../input/file")

    def test_size_guard_and_symlink_directories(self):
        (self.root / "output" / "large").write_bytes(b"abc")
        self.sandbox.config = SandboxConfig(max_workspace_bytes=2)
        with self.assertRaisesRegex(SandboxError, "limit"):
            self.sandbox._check_workspace()
        (self.root / "scripts").rmdir()
        (self.root / "scripts").symlink_to(self.root / "input", target_is_directory=True)
        with self.assertRaisesRegex(SandboxError, "real task directory"):
            DockerSandbox(SandboxConfig(), self.root)

    def test_cleanup_failure_is_visible_and_blocks_reuse(self):
        self.sandbox._active.add("owned")
        with patch.object(self.sandbox, "_control", return_value=subprocess.CompletedProcess([], 1, "", "daemon unavailable")):
            with self.assertRaises(SandboxError):
                self.sandbox.close()
        self.assertTrue(self.sandbox.broken)
        self.assertEqual(self.sandbox.metadata()["active_containers"], ["owned"])
        with self.assertRaises(SandboxError):
            self.sandbox.execute("echo unsafe")

    def test_close_is_idempotent_and_only_removes_owned_containers(self):
        self.sandbox._active.add("owned")
        with patch.object(self.sandbox, "_control", return_value=subprocess.CompletedProcess([], 0, "", "")) as control:
            self.sandbox.close()
            self.sandbox.close()
            control.assert_called_once_with(["rm", "--force", "owned"], check=False)

    def test_create_failure_still_cleans_and_records_failure(self):
        self.sandbox.image_id = "fixed"
        with patch.object(self.sandbox, "_control") as control:
            control.side_effect = [SandboxError("create timed out"), subprocess.CompletedProcess([], 0, "", "")]
            with self.assertRaisesRegex(SandboxError, "create timed out"):
                self.sandbox.execute("echo hello")
        self.assertEqual(self.sandbox.executions[0]["cleanup_status"], "removed")
        self.assertFalse(self.sandbox._active)

    def test_interrupt_cleans_container(self):
        self.sandbox.image_id = "fixed"
        with patch.object(self.sandbox, "_control") as control:
            control.side_effect = [KeyboardInterrupt(), subprocess.CompletedProcess([], 0, "", "")]
            with self.assertRaises(KeyboardInterrupt):
                self.sandbox.execute("echo hello")
        self.assertFalse(self.sandbox._active)
        self.assertEqual(self.sandbox.executions[0]["cleanup_status"], "removed")

    def test_container_that_never_started_is_not_reported_as_success(self):
        self.sandbox.image_id = "fixed"
        self.sandbox._docker = "/docker"
        self.sandbox.context = "local"
        process = Mock()
        process.poll.return_value = 1
        with patch.object(self.sandbox, "_control") as control, \
                patch("bpcagent.sandbox.docker.subprocess.Popen", return_value=process), \
                patch.object(self.sandbox, "_collect_output", return_value=([b"", b"start failed"], [False, False], False)):
            control.side_effect = [
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 0, json.dumps({"Status": "created", "Running": False, "ExitCode": 0}), ""),
                subprocess.CompletedProcess([], 0, "", ""),
            ]
            with self.assertRaisesRegex(SandboxError, "did not finish"):
                self.sandbox.execute("echo hello")
        self.assertTrue(self.sandbox.broken)
        self.assertEqual(self.sandbox.executions[0]["cleanup_status"], "removed")

    def test_tool_keeps_command_only_schema_and_error_contract(self):
        backend = Mock(spec=DockerSandbox)
        registry = ToolRegistry()
        registry.register_tool(SandboxShellTool(backend))
        backend.execute.return_value = {"exit_code": 0, "stdout": "ok"}
        self.assertEqual(registry.execute_tool("run_shell", {"command": "echo ok"}).data["stdout"], "ok")
        invalid = registry.execute_tool("run_shell", {"command": "echo ok", "mount": "/"})
        self.assertEqual(invalid.error.code, "invalid_arguments")
        self.assertEqual(backend.execute.call_count, 1)
        backend.execute.side_effect = SandboxError("daemon offline")
        self.assertEqual(registry.execute_tool("run_shell", {"command": "echo ok"}).error.code, "execution_error")


if __name__ == "__main__":
    unittest.main()
