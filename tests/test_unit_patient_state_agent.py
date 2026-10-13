"""Test the packaged extraction/normalization workflow with real shell calls."""

import json
import os
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bpcagent import Config, LLMResponse, ToolCall, UnitPatientStateAgent
from bpcagent.agents import UnitPatientStateAgent as ExportedAgent
from bpcagent.tools.bpc_patient import PatientArtifactStore, PatientEvidenceTool, PatientStateTool


def tool_response(name, arguments, call_id):
    return LLMResponse(tool_calls=[ToolCall(
        id=call_id, name=name, arguments=json.dumps(arguments),
    )], finish_reason="tool_calls")


@unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "POSIX shell required")
class UnitPatientStateAgentTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        previous = Path.cwd()
        os.chdir(temp.name)
        self.addCleanup(os.chdir, previous)
        self.work_dir = Path(temp.name) / "workspace"
        self.state = {
            "meta": {"patient_id": "P1", "cutoff_day": 220},
            "evidence_refs": {"E-test": {}},
            "observations": {"tumor_markers": [
                {"tumor_marker_id": "M1", "analyte_raw": "PSA", "unit_raw": "ug/L", "value_raw": "0"},
                {"tumor_marker_id": "M2", "analyte_raw": "testosterone", "unit_raw": "nmol/L", "value_raw": "1"},
                {"tumor_marker_id": "M3", "analyte_raw": "PSA", "unit_raw": "nmol/L", "value_raw": "2"},
            ]},
        }
        bound = SimpleNamespace(
            patient_id="P1", cancer_id=None, max_cutoff_day=220,
            get_patient_state=Mock(return_value=self.state),
            get_evidence=Mock(return_value={"E-test": {"sources": []}}),
        )
        store = PatientArtifactStore(self.work_dir)
        tools = (PatientStateTool(bound, store), PatientEvidenceTool(bound, store))
        self.llm = Mock()
        with patch("bpcagent.agents.patient_state.create_patient_tools", return_value=tools):
            self.agent = UnitPatientStateAgent(
                patient_id="P1", cutoff_day=220, data_dir="unused", work_dir=self.work_dir,
                config=Config(max_tool_iterations=4), llm=self.llm,
            )
        self.extraction_registry = self.agent.tool_registry
        self.extraction = [
            tool_response("get_patient_state", {"cutoff_day": 220}, "state"),
            tool_response("get_patient_evidence", {"cutoff_day": 220, "evidence_refs": ["E-test"]}, "evidence"),
        ]

    def shell_response(self, *, incorrect=False, tamper=False):
        normalizations = [
            {"status": "normalized", "value": 0, "unit": "ng/mL", "factor": 1, "reason": None},
            {"status": "normalized", "value": 999 if incorrect else 28.842,
             "unit": "ng/dL", "factor": 28.842, "reason": None},
            {"status": "unresolved", "value": None, "unit": None, "factor": None,
             "reason": "Unsupported unit for PSA."},
        ]
        code = (
            "import json; from pathlib import Path; "
            "p=Path('patient_state_day_220_001.json'); state=json.loads(p.read_text()); "
            f"norms={normalizations!r}; "
            "markers=[dict(marker, normalization=norm) for marker,norm "
            "in zip(state['observations']['tumor_markers'], norms)]; "
            "out={'source_state':p.name, 'meta':state['meta'], 'tumor_markers':markers}; "
            "target=Path('tumor_markers_standardized.json'); "
            "target.write_text(json.dumps(out)); json.loads(target.read_text())"
        )
        if tamper:
            code += "; p.write_text('{}')"
        return tool_response("run_shell", {
            "command": shlex.quote(sys.executable) + " -c " + shlex.quote(code),
        }, "shell")

    def report(self):
        return LLMResponse(content="Normalized: 2. Unresolved: 1. Output: tumor_markers_standardized.json.", finish_reason="stop")

    def test_public_agent_runs_both_stages_and_saves_final_log(self):
        self.assertIs(ExportedAgent, UnitPatientStateAgent)
        self.llm.invoke.side_effect = self.extraction + [self.shell_response(), self.report()]
        result = self.agent.run()
        self.assertTrue(result.finished)
        self.assertEqual(result.info["validation_status"], "passed")
        self.assertEqual(self.llm.invoke.call_count, 4)
        self.assertIs(self.agent.tool_registry, self.extraction_registry)
        self.assertEqual(json.loads((self.work_dir / "patient_state_day_220_001.json").read_text()), self.state)
        self.assertFalse((self.work_dir / "answer.json").exists())
        log = json.loads(self.agent.log_path.read_text())
        self.assertTrue(self.agent.log_path.name.startswith("UnitPatientStateAgent_"))
        self.assertTrue(log["finished"])
        self.assertEqual(log["validation_attempts"], [{"turn": 2, "status": "passed"}])
        processing = self.llm.invoke.call_args_list[2]
        self.assertEqual([tool["function"]["name"] for tool in processing.kwargs["tools"]], ["run_shell"])
        self.assertIn("Use English", processing.args[0][0]["content"])

    def test_bad_conversion_is_retried_and_corrected(self):
        self.llm.invoke.side_effect = self.extraction + [
            self.shell_response(incorrect=True), self.report(), self.shell_response(), self.report(),
        ]
        result = self.agent.run()
        self.assertTrue(result.finished)
        self.assertEqual([item["status"] for item in self.agent.validation_attempts], ["failed", "passed"])
        self.assertIn("incorrect conversion", self.agent.validation_attempts[0]["error"])
        feedback = [m["content"] for m in result.transcript if m["role"] == "user"]
        self.assertTrue(any("Independent validation failed" in text for text in feedback))

    def test_original_file_changes_fail_validation_and_are_logged(self):
        self.llm.invoke.side_effect = self.extraction + [self.shell_response(tamper=True), self.report()]
        result = self.agent.run(max_tool_iterations=2)
        self.assertFalse(result.finished)
        self.assertEqual(result.info["stop_reason"], "unit_processing_failed")
        self.assertIn("Original extracted file was modified", result.info["error"])
        self.assertFalse(json.loads(self.agent.log_path.read_text())["finished"])
        self.assertIs(self.agent.tool_registry, self.extraction_registry)

    def test_text_without_shell_execution_cannot_complete_processing(self):
        self.llm.invoke.side_effect = self.extraction + [self.report(), self.report()]
        result = self.agent.run(max_tool_iterations=2)
        self.assertFalse(result.finished)
        self.assertIn("No successful shell call", result.info["error"])

    def test_failed_extraction_does_not_start_processing(self):
        self.llm.invoke.return_value = self.report()
        result = self.agent.run()
        self.assertFalse(result.finished)
        self.assertEqual(result.info["stop_reason"], "incomplete_extraction")
        self.assertEqual(result.info["validation_status"], "not_run")
        self.assertEqual(self.llm.invoke.call_count, 1)
        self.assertFalse((self.work_dir / "tumor_markers_standardized.json").exists())


if __name__ == "__main__":
    unittest.main()
