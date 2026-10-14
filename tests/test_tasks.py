"""Task contracts, independent runs and artifact discovery without model calls."""

from datetime import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from bpcagent import AgentResult, Config, PatientStateTask, PatientStateTaskConfig, Task, TaskResult
from bpcagent.agents.exceptions import ConfigException
from bpcagent.tasks.base import Task as BaseTask, TaskResult as BaseTaskResult


class TaskTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.data = self.root / "dataset"
        self.data.mkdir()
        self.rules = self.root / "rules.json"
        self.rules.write_text("{}")
        self.config = PatientStateTaskConfig(data_dir=self.data, rules_path=self.rules,
                                            workspace_root=self.root / "workspace")
        self.task = PatientStateTask(self.config)

    def fake_agent(self, *, finished=True, artifacts=True, partial=False, **kwargs):
        root = kwargs["work_dir"]
        for name in ("input", "scripts", "output", "tmp"):
            (root / name).mkdir()
        if artifacts or partial:
            (root / "input" / f"patient_state_day_{kwargs['cutoff_day']}_001.json").write_text("{}")
        if artifacts:
            (root / "input" / f"patient_evidence_day_{kwargs['cutoff_day']}_002.json").write_text("{}")
            (root / "scripts" / "standardize_tm.py").write_text("# synthetic fixture")
            (root / "output" / "tumor_markers_standardized.json").write_text("{}")
        log = self.root / f"{root.name}.json"
        log.write_text("{}")
        info = {} if finished is True else {"stop_reason": "unit_processing_failed", "error": "synthetic failure"}
        return SimpleNamespace(log_path=log, run=Mock(return_value=AgentResult(
            finished=finished, output={"text": "completed"}, info=info)))

    def test_base_is_abstract_and_result_containers_are_independent(self):
        self.assertIs(Task, BaseTask)
        self.assertIs(TaskResult, BaseTaskResult)
        with self.assertRaises(TypeError):
            Task()
        first, second = TaskResult(), TaskResult()
        first.artifacts["x"] = self.root
        first.info["error"] = "failed"
        self.assertEqual(second.artifacts, {})
        self.assertEqual(second.info, {})
        self.assertFalse(second.finished)

    def test_query_creates_expected_agent_and_directory(self):
        with patch("bpcagent.tasks.patient_state.datetime") as clock, \
                patch("bpcagent.tasks.patient_state.UnitPatientStateAgent", side_effect=self.fake_agent) as factory:
            clock.now.return_value = datetime(2026, 10, 14, 9, 30, 5)
            result = self.task.run(patient_id="GENIE-P1", cutoff_day=220, cancer_id="C1")
        self.assertTrue(result.finished)
        self.assertEqual(result.task_dir.name, "20261014_093005_GENIE-P1_220")
        self.assertEqual(str(clock.now.call_args.args[0]), "Asia/Shanghai")
        arguments = factory.call_args.kwargs
        self.assertEqual(arguments["patient_id"], "GENIE-P1")
        self.assertEqual(arguments["cutoff_day"], 220)
        self.assertEqual(arguments["cancer_id"], "C1")
        self.assertEqual(arguments["data_dir"], self.data)
        self.assertEqual(arguments["rules_path"], self.rules)
        self.assertIs(arguments["sandbox_config"], self.task.config.sandbox_config)
        self.assertEqual(set(result.artifacts), {"state", "evidence", "normalized", "script"})
        self.assertTrue(all(p.is_absolute() and p.is_file() for p in result.artifacts.values()))
        self.assertTrue(result.log_path.is_absolute())
        self.assertFalse((result.task_dir / "run_result.json").exists())

    def test_repeated_task_object_uses_fresh_agents_and_preserves_previous_result(self):
        agents = []

        def create(**kwargs):
            agent = self.fake_agent(**kwargs)
            agents.append(agent)
            return agent

        with patch("bpcagent.tasks.patient_state.UnitPatientStateAgent", side_effect=create):
            first = self.task.run(patient_id="P1", cutoff_day=0)
            second = self.task.run(patient_id="P2", cutoff_day=-1)
        self.assertNotEqual(first.task_dir, second.task_dir)
        self.assertIsNot(agents[0], agents[1])
        self.assertIsNot(first.agent_result, second.agent_result)
        self.assertTrue(all(p.exists() for p in first.artifacts.values()))
        self.assertIsNot(first.artifacts, second.artifacts)

    def test_same_second_collision_never_overwrites_or_constructs_another_agent(self):
        with patch("bpcagent.tasks.patient_state.datetime") as clock, \
                patch("bpcagent.tasks.patient_state.UnitPatientStateAgent", side_effect=self.fake_agent) as factory:
            clock.now.return_value = datetime(2026, 10, 14, 9, 30, 5)
            first = self.task.run(patient_id="P1", cutoff_day=1)
            before = first.artifacts["state"].read_bytes()
            with self.assertRaises(FileExistsError):
                self.task.run(patient_id="P1", cutoff_day=1)
            self.assertEqual(first.artifacts["state"].read_bytes(), before)
            self.assertEqual(factory.call_count, 1)

    def test_bad_query_has_no_workspace_side_effects(self):
        invalid = [
            {"patient_id": value, "cutoff_day": 1}
            for value in ("", " ", "..", "../P1", "P1/child", "P1\\child", "P1\n", "P1,*", 1)
        ] + [{"patient_id": "P1", "cutoff_day": value} for value in (True, 1.5, "220")]
        invalid.append({"patient_id": "P1", "cutoff_day": 1, "cancer_id": " "})
        with patch("bpcagent.tasks.patient_state.UnitPatientStateAgent") as factory:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.task.run(**kwargs)
            factory.assert_not_called()
        self.assertFalse(self.config.workspace_root.exists())

    def test_agent_failure_returns_partial_artifacts_and_original_error(self):
        with patch("bpcagent.tasks.patient_state.UnitPatientStateAgent",
                   side_effect=lambda **kwargs: self.fake_agent(finished=False, artifacts=False, partial=True, **kwargs)):
            result = self.task.run(patient_id="P1", cutoff_day=1)
        self.assertFalse(result.finished)
        self.assertEqual(set(result.artifacts), {"state"})
        self.assertEqual(result.info["stop_reason"], "unit_processing_failed")
        result.info["error"] = "task note"
        self.assertEqual(result.agent_result.info["error"], "synthetic failure")

    def test_existing_files_do_not_turn_failed_or_unknown_agent_result_into_success(self):
        for finished in (False, None):
            with self.subTest(finished=finished), patch("bpcagent.tasks.patient_state.UnitPatientStateAgent",
                    side_effect=lambda **kwargs: self.fake_agent(finished=finished, **kwargs)):
                result = self.task.run(patient_id=f"P-{finished}", cutoff_day=1)
                self.assertFalse(result.finished)
                self.assertEqual(len(result.artifacts), 4)

    def test_missing_artifacts_do_not_change_original_agent_result(self):
        with patch("bpcagent.tasks.patient_state.UnitPatientStateAgent",
                   side_effect=lambda **kwargs: self.fake_agent(artifacts=False, **kwargs)):
            result = self.task.run(patient_id="P1", cutoff_day=1)
        self.assertFalse(result.finished)
        self.assertTrue(result.agent_result.finished)
        self.assertEqual(result.agent_result.info, {})
        self.assertEqual(result.info["stop_reason"], "incomplete_artifacts")

    def test_programming_and_configuration_errors_propagate(self):
        for error in (ConfigException("invalid model"), RuntimeError("unexpected bug")):
            with self.subTest(error=error), patch("bpcagent.tasks.patient_state.UnitPatientStateAgent", side_effect=error):
                with self.assertRaises(type(error)):
                    self.task.run(patient_id=type(error).__name__, cutoff_day=1)

    def test_invalid_paths_timezone_and_sandbox_disabled_are_rejected(self):
        with self.assertRaises(NotADirectoryError):
            PatientStateTask(self.config.model_copy(update={"data_dir": self.root / "missing"}))
        with self.assertRaises(FileNotFoundError):
            PatientStateTask(self.config.model_copy(update={"rules_path": self.root / "missing.json"}))
        for values in ({"timezone": "not/a/timezone"}, {"sandbox_config": None}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                PatientStateTaskConfig(data_dir=self.data, workspace_root=self.root / "work", **values)

    def test_paths_and_environment_are_snapshotted_once(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        config = PatientStateTaskConfig(data_dir="dataset", workspace_root="work", rules_path="rules.json")
        with patch.dict(os.environ, {"LLM_MODEL_ID": "first", "TEMPERATURE": "0.1"}):
            task = PatientStateTask(config)
        os.chdir(self.data)
        with patch.dict(os.environ, {"LLM_MODEL_ID": "changed", "TEMPERATURE": "bad"}), \
                patch("bpcagent.tasks.patient_state.UnitPatientStateAgent", side_effect=self.fake_agent) as factory:
            result = task.run(patient_id="P1", cutoff_day=1)
        self.assertEqual(result.task_dir.parent, self.root / "work")
        snapshot = factory.call_args.kwargs["config"]
        self.assertEqual(snapshot.model, "first")
        self.assertEqual(snapshot.temperature, 0.1)
        self.assertEqual(Config.resolve(snapshot).model, "first")
        self.assertIsNone(config.agent_config.model)

    def test_collects_latest_sequence_and_excludes_links_or_unrelated_files(self):
        for name in ("input", "scripts", "output"):
            (self.root / name).mkdir()
        for name in ("patient_state_day_1_999.json", "patient_state_day_1_1000.json",
                     "patient_state_day_2_1001.json", "patient_state_day_1_bad.json"):
            (self.root / "input" / name).write_text("{}")
        (self.root / "scripts" / "standardize_tm.py").symlink_to(self.rules)
        (self.root / "output" / "tumor_markers_standardized.json").symlink_to(self.rules)
        artifacts = self.task._collect_artifacts(self.root, 1)
        self.assertEqual(set(artifacts), {"state"})
        self.assertEqual(artifacts["state"].name, "patient_state_day_1_1000.json")


if __name__ == "__main__":
    unittest.main()
