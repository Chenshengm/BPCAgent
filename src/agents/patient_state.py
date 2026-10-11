"""调用工具提取患者数据；完成后直接停止，不请求模型分析。"""

from pathlib import Path
import json

from .agents import BasicAgent
from .config import Config
from .exceptions import AgentException, ConfigException
from .messages import Message
from .models import MyLLM
from .result import AgentResult
from ..tools import ToolRegistry
from ..tools.bpc_patient import create_patient_tools


class PatientStateAgent(BasicAgent):
    """绑定一个患者和提取日期，通过两个工具将数据写入工作目录。"""

    def __init__(
        self,
        *,
        patient_id: str,
        cutoff_day: int,
        data_dir: str | Path,
        work_dir: str | Path,
        rules_path: str | Path | None = None,
        cancer_id: str | None = None,
        config: Config | None = None,
        llm: MyLLM | None = None,
    ):
        self.patient_id = patient_id
        self.cutoff_day = cutoff_day
        self.work_dir = Path(work_dir).resolve()
        registry = ToolRegistry()
        for tool in create_patient_tools(
            patient_id=patient_id, max_cutoff_day=cutoff_day,
            data_dir=data_dir, work_dir=self.work_dir,
            rules_path=rules_path, cancer_id=cancer_id,
        ):
            registry.register_tool(tool)
        super().__init__(
            name="PatientStateAgent", llm=llm, config=config, tool_registry=registry,
            sys_prompt=(
                "You are a patient data extraction agent. Use tools only to retrieve and save data. "
                "First call get_patient_state for the specified cutoff day. "
                "Then select at least one evidence reference from the result and call get_patient_evidence with the same cutoff day. "
                "The tools save JSON files automatically; no separate file-writing step is needed. "
                "Issue tool calls only. Do not summarize the patient's condition, analyze findings, infer missing facts, or recommend treatment. "
                "If a tool fails, correct the call using the error details. Do not report failure as completion. Use English for any generated text."
            ),
        )

    def run(self, input: str | None = None, **kwargs) -> AgentResult:
        # 每次都是独立的提取任务，不使用之前的对话作为完成依据。
        self.clear_history()
        task = f"Extract and save the state and evidence for patient {self.patient_id} through day {self.cutoff_day}."
        if input:
            task += f"\nAdditional execution requirements: {input}"
        return super().run(task, **kwargs)

    def _run_tool_loop(self, messages: list, input: str, max_tool_iterations: int, **kwargs) -> str:
        """复用框架的模型响应校验、工具执行、失败结果和日志保存。"""
        if "tools" in kwargs:
            raise ConfigException("Agent tools must come from the tool registry")
        options = {**kwargs, "tools": self.tool_registry.to_openai_tools()}
        state_saved = False
        evidence_saved = False
        for _ in range(max_tool_iterations):
            response = self._invoke_response(messages, **options)
            if not response.tool_calls:
                raise AgentException(
                    "The model did not complete state and evidence extraction for the specified day. A text response does not count as completion.",
                    code="incomplete_extraction",
                )
            for call in response.tool_calls:
                result = self._execute_tool_call(call)
                messages.append(Message(result.model_dump_json(), "tool", tool_call_id=call.id).to_dict())
                if result.status != "success":
                    continue
                arguments = json.loads(call.arguments)
                if arguments.get("cutoff_day") != self.cutoff_day:
                    continue
                if call.name == "get_patient_state":
                    state_saved = True
                elif call.name == "get_patient_evidence" and state_saved:
                    evidence_saved = True
            if state_saved and evidence_saved:
                # 这是程序执行状态；不追加模型请求或伪造 assistant 总结消息。
                return f"Patient data extraction completed. State and evidence saved to: {self.work_dir}"
        raise AgentException(
            "Model iteration limit reached before state and evidence extraction completed.",
            code="step_limit_exceeded",
        )
