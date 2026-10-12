"""患者数据提取流程：只按工具执行结果判断完成，不生成分析。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bpcagent import Config, LLMResponse, PatientStateAgent, ToolCall
from bpcagent.tools.bpc_patient import PatientArtifactStore, PatientEvidenceTool, PatientStateTool


def tool_response(name, arguments, call_id):
    return LLMResponse(tool_calls=[ToolCall(
        id=call_id, name=name, arguments=arguments,
    )], finish_reason="tool_calls")


def state_response(day=220):
    return tool_response("get_patient_state", f'{{"cutoff_day": {day}}}', "state")


def evidence_response(day=220):
    return tool_response("get_patient_evidence", f'{{"cutoff_day": {day}, "evidence_refs": ["E-test"]}}', "evidence")


class PatientStateAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.bound = SimpleNamespace(
            patient_id="P1", cancer_id=None, max_cutoff_day=220,
            get_patient_state=Mock(return_value={"evidence_refs": {"E-test": {}}}),
            get_evidence=Mock(return_value={"E-test": {"sources": []}}),
        )
        store = PatientArtifactStore(self.directory)
        tools = (PatientStateTool(self.bound, store), PatientEvidenceTool(self.bound, store))
        self.llm = Mock()
        with patch("bpcagent.agents.patient_state.create_patient_tools", return_value=tools):
            self.agent = PatientStateAgent(
                patient_id="P1", cutoff_day=220, data_dir="unused", work_dir=self.directory,
                llm=self.llm, config=Config.resolve(environ={}, overrides={"max_tool_iterations": 3}),
            )
        log_patcher = patch.object(self.agent, "_save_history")
        self.log = log_patcher.start()
        self.addCleanup(log_patcher.stop)

    def test_stops_after_two_tools_without_analysis_request(self):
        self.llm.invoke.side_effect = [state_response(), evidence_response()]
        result = self.agent.run()
        self.assertTrue(result.finished)
        self.assertEqual(self.llm.invoke.call_count, 2)
        self.assertIn("Patient data extraction completed", result.output["text"])
        self.assertEqual(result.transcript[-1]["role"], "tool")
        self.assertEqual(len(list(self.directory.glob("*.json"))), 2)
        self.assertFalse((self.directory / "answer.json").exists())
        self.log.assert_called_once()

    def test_text_answer_is_not_successful_extraction(self):
        self.llm.invoke.return_value = LLMResponse(content="Here is a clinical analysis", finish_reason="stop")
        result = self.agent.run()
        self.assertFalse(result.finished)
        self.assertEqual(result.output, {})
        self.assertEqual(result.info["stop_reason"], "incomplete_extraction")
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_extraction_preserves_reasoning_between_tool_calls(self):
        state = state_response().model_copy(update={"reasoning_content": "先提取 state。"})
        evidence = evidence_response().model_copy(update={"reasoning_content": "再提取 evidence。"})

        def invoke(messages, **kwargs):
            if self.llm.invoke.call_count == 1:
                return state
            self.assertEqual(messages[-2]["reasoning_content"], state.reasoning_content)
            return evidence

        self.llm.invoke.side_effect = invoke
        result = self.agent.run()
        self.assertTrue(result.finished)
        self.assertEqual([m["reasoning_content"] for m in result.transcript
                          if m["role"] == "assistant"],
                         [state.reasoning_content, evidence.reasoning_content])

    def test_failed_tool_can_be_corrected_before_completion(self):
        self.bound.get_evidence.side_effect = [OSError("write failed"), {"E-test": {}}]
        self.llm.invoke.side_effect = [state_response(), evidence_response(), evidence_response()]
        result = self.agent.run()
        self.assertTrue(result.finished)
        self.assertEqual(self.llm.invoke.call_count, 3)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 2)

    def test_other_date_does_not_complete_target_task(self):
        self.llm.invoke.side_effect = [state_response(219), evidence_response(219)]
        result = self.agent.run(max_tool_iterations=2)
        self.assertFalse(result.finished)
        self.assertEqual(result.info["stop_reason"], "step_limit_exceeded")

    def test_previous_run_does_not_count_as_current_completion(self):
        self.llm.invoke.side_effect = [state_response(), evidence_response()]
        self.assertTrue(self.agent.run().finished)
        self.llm.invoke.side_effect = [LLMResponse(content="Done", finish_reason="stop")]
        self.assertFalse(self.agent.run().finished)


if __name__ == "__main__":
    unittest.main()
