# BPCAgent

一个用于注册工具、创建智能体并执行工具调用的 Python 框架。完整安装、配置与接口说明见 [doc/detail.md](doc/detail.md)。

以下示例在 `BPCAgent` 目录运行，要求已安装 `bpcagent` 和示例依赖 `python-dotenv`；真实模型的配置写入 `.env`：`LLM_API_KEY`、`LLM_MODEL_ID`、`LLM_BASE_URL`。模型服务需支持工具调用。

## 已实现的智能体与工具

| 智能体 | 用途 | 调用入口 |
| --- | --- | --- |
| `BasicAgent` | 通用问答与工具调用，保留对话历史 | `agent.run("Task")` |
| `PatientStateAgent` | 提取并保存指定患者、截止日期的状态与证据，完成后直接停止，不生成模型分析 | `agent.run()` |
| `UnitPatientStateAgent` | 提取后通过 Shell 核查单位、根据证据直接应用推断修正并统一单位，报告改动位置与理由，独立验收后返回报告 | `agent.run()` |

| 工具类 | 注册名称 | 参数与行为 |
| --- | --- | --- |
| `CalculatorTool` | `calculate` | `input`：数学表达式，返回计算结果 |
| `ShellTool` | `run_shell` | `command`：在固定工作目录执行命令，读取／写入文件并返回退出码、输出和超时信息 |
| `PatientStateTool` | `get_patient_state` | `cutoff_day`：截止日期，返回患者状态并保存 JSON |
| `PatientEvidenceTool` | `get_patient_evidence` | `cutoff_day`、`evidence_refs`：返回对应证据并保存 JSON |

患者工具通过 `create_patient_tools()` 创建；患者、数据目录、日期上限与工作目录由应用绑定，模型只需填写查询参数。

## 通过 Task 运行完整患者流程

`PatientStateTask` 将配置、工作目录创建、Agent 调用和产物收集封装为任务入口。先用 `PatientStateTaskConfig` 配置数据路径、总工作区、模型和 sandbox，再调用 `task.run(patient_id=..., cutoff_day=..., cancer_id=...)`；其中 `cancer_id` 可省略。每次调用创建独立 Agent，返回 `TaskResult`，可直接读取 `finished`、`artifacts`、`task_dir`、`log_path` 和 `agent_result`。

抽象 `Task` 与 `TaskResult` 同在 `src/tasks/base.py`；现有 Agent 的执行逻辑保持不变。完整示例和参数说明见 [Task 使用文档](doc/tasks.md)。

## 注册工具、创建并使用智能体

```python
from dotenv import load_dotenv
from bpcagent import BasicAgent, CalculatorTool, Config, ToolRegistry

load_dotenv(".env", override=False)
config = Config(max_tool_iterations=6)
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

agent = BasicAgent(
    name="CalculatorDemo",
    config=config,
    tool_registry=registry,
    sys_prompt="Use the calculate tool for arithmetic. Respond in English.",
)
result = agent.run("Calculate (12 + 8) * 3 using the tool.")
if result.finished:
    print(result.output["text"])
else:
    print("Run incomplete:", result.info)
```

也可以绕过模型直接调用已注册工具；`ToolResult` 的 `status` 为 `success` 或 `error`，结果在 `data`，错误在 `error`：

```python
result = registry.execute_tool("calculate", {"input": "2 + 3 * 4"})
print(result.data if result.status == "success" else result.error)
```

## 使用 Shell 工具

`ShellTool` 支持 macOS、Linux，在本机以当前用户权限执行命令。每次启动独立的非交互 shell：文件保留，`cd` 和 `export` 不跨调用保留。工作目录不是沙箱边界，工具不限制网络或目录外访问；实验请使用独立目录和合成数据。

```python
from pathlib import Path
from bpcagent import ShellTool, ShellToolConfig, ToolRegistry

work_dir = Path("/tmp/bpc-shell-demo")
work_dir.mkdir(parents=True, exist_ok=True)
registry = ToolRegistry()
registry.register_tool(ShellTool(ShellToolConfig(work_dir=work_dir)))
result = registry.execute_tool("run_shell", {
    "command": "printf 'hello\\n' > result.txt\ncat result.txt",
})
print(result.data if result.status == "success" else result.error)
```

应用通过 `ShellToolConfig` 配置 `work_dir`、`shell_path`（默认 `/bin/bash`）、`timeout_seconds`（30 秒）、`termination_grace_seconds`（2 秒）、`max_output_bytes`（每条输出流 32768 字节）和 `env_overrides`。子进程不自动继承模型密钥等主进程环境变量；如需虚拟环境的 Python，请配置 `PATH` 或在命令中使用解释器绝对路径。

结果包含 `exit_code`、`stdout`、`stderr`、`timed_out`、`duration_ms`、`stdout_truncated`、`stderr_truncated` 和 `cwd`。`ToolResult.status="success"` 表示成功取得执行结果，还需检查退出码和超时标记。非零退出码和超时不会触发自动重试，已写入的文件不会回滚。超时清理包括终止宽限期，因此实际耗时可以略长于配置时限。

只保留 stdout、stderr 的前 N 字节，超限部分继续读取但不保留；UTF-8 非法或被截断的字节使用替换字符。需要大量结果时直接写入文件，再按需读取。不要运行交互程序、后台服务或主动脱离进程组的任务。

独立实验复用 `BasicAgent`，不调用 PatientStateAgent：

```bash
# 模型响应预设，但 shell、脚本和文件读写实际执行；不访问模型服务
python examples/shell_experiment.py

# 使用 BPCAgent/.env 中的模型配置，真实验证多轮文件处理
python examples/shell_experiment.py --live
```

每次创建唯一实验目录（默认系统临时目录，可用 `--work-dir` 指定父目录），保留合成输入、输出和 `agent_result.json`。任务统计空字段，零值不算缺失；示例独立校验输出文件和原输入未变，模型仅声称完成不会通过。Agent 调用轨迹还会按现有机制写入 `log/`。

运行离线测试：`python -B -m unittest discover -s tests -v`。

## 使用患者数据提取智能体

还需在同一 Python 环境安装 `Pathway-Agent` 提供的 `bpc_state` 包，并准备数据集。以下路径按 `code/BPCAgent` 与 `code/Pathway-Agent` 并列设置，每次运行创建独立的工作目录。

```python
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
from bpcagent import Config, PatientStateAgent

load_dotenv(".env", override=False)
root = Path.cwd().parent
pathway = root / "Pathway-Agent"
work_dir = root / "workspace" / f"run_{uuid4().hex}"

agent = PatientStateAgent(
    patient_id="GENIE-DFCI-051542",
    cutoff_day=220,
    data_dir=pathway / "dataset",
    rules_path=pathway / "config" / "field_rules.json",
    work_dir=work_dir,
    config=Config(max_tool_iterations=6),
)
result = agent.run()
if result.finished:
    print(result.output["text"])
else:
    print("Extraction incomplete:", result.info)
```

智能体先调用 `get_patient_state`，再使用返回的证据引用调用 `get_patient_evidence`，两者使用指定的同一截止日期。成功提取状态与至少一项证据后，程序直接返回完成状态；这不代表导出了全部证据。

工具自动在 `work_dir` 保存 `patient_state_day_220_001.json`、`patient_evidence_day_220_002.json` 等文件，序号随保存递增，不创建 `answer.json`。Agent 调用日志另存于当前运行目录的 `log/`。

### 提取并统一单位

`UnitPatientStateAgent` 继承 `PatientStateAgent`，支持相同的患者构造参数。它需要在 WSL、Linux 或 macOS 下运行；复用上例的环境配置和路径即可创建：

```python
from bpcagent import UnitPatientStateAgent

agent = UnitPatientStateAgent(
    patient_id="GENIE-DFCI-051542",
    cutoff_day=220,
    data_dir=pathway / "dataset",
    rules_path=pathway / "config" / "field_rules.json",
    work_dir=root / "workspace" / f"run_{uuid4().hex}",
    config=Config(max_tool_iterations=10),
)
result = agent.run()
print(result.output["text"] if result.finished else result.info)
print("Full log:", agent.log_path)
```

两个阶段各自使用 `max_tool_iterations` 轮次上限。标准化结果保存为 `tumor_markers_standardized.json`（schema `2.1`），保留原始字段，新增 `unit_assessment` 和 `normalization`。智能体可以根据当前患者的可见证据直接修正单位或补充总／游离睾酮类型，标记为 `inferred_correction` 并应用到标准化结果，无需外部确认。证据仍不足时才保留待核验。可选的 `unit_corrections_path` 用于提供优先采用的已确认资料；模型推断不会冒充 `confirmed_correction`。宿主将改动位置、前后值、理由和证据整理到 `info.unit_changes`，并追加到最终文字报告。

输入目录新增 `unit_inputs.json`，保存固定规则和仅适用于当前患者、截止日、记录的纠错资料。模型必须保存 `standardize_tm.py`，在独立 shell 调用中执行提示词给出的完整命令；宿主程序检查执行成功、脚本内容未变及换算结果。验收失败会反馈模型修复。`result.info["unit_summary"]` 是宿主验收后的分类计数，存在 `unresolved` 不等于执行失败。日志仍为 `log/UnitPatientStateAgent_*.json`，包含规则版本、验收记录和最终状态。协议、纠错资料示例与限制见 [Task 使用文档](doc/tasks.md)。

如需隔离执行，额外传入 `sandbox_config=SandboxConfig(image="bpcagent-sandbox:0.1")`（从 `bpcagent` 导入 `SandboxConfig`）。需先启动本机 Docker 并构建镜像。此模式中 `work_dir` 参数是整个任务目录：提取文件位于 `input/`，脚本位于 `scripts/`，标准化结果位于 `output/`，临时文件位于 `tmp/`。输入对容器只读，每次 shell 调用使用独立的无网络容器。详见 [sandbox 配置、构建和测试说明](doc/sandbox.md)。未传入配置时保留本地执行方式；sandbox 失败不会退回本地执行。

### 单独调用患者工具

不需要智能体时，复用上例的 `root`、`pathway`，用工厂创建工具并直接执行；此过程不调用模型：

```python
from bpcagent import ToolRegistry
from bpcagent.tools.bpc_patient import create_patient_tools

registry = ToolRegistry()
for tool in create_patient_tools(
    patient_id="GENIE-DFCI-051542",
    max_cutoff_day=220,
    data_dir=pathway / "dataset",
    rules_path=pathway / "config" / "field_rules.json",
    work_dir=root / "workspace" / f"run_{uuid4().hex}",
):
    registry.register_tool(tool)

state = registry.execute_tool("get_patient_state", {"cutoff_day": 220})
if state.status != "success":
    raise RuntimeError(state.error)
refs = list(state.data["evidence_refs"])
if refs:
    evidence = registry.execute_tool("get_patient_evidence", {
        "cutoff_day": 220, "evidence_refs": refs[:1],
    })
    print("Evidence extraction:", evidence.status)
else:
    print("No evidence references available.")
```

也可把此注册表传给 `BasicAgent`。查询日期不能超过 `max_cutoff_day`；证据引用必须来自同一截止日期的状态结果。

## 创建新工具

在 `src/tools/uppercase.py` 定义参数模型和工具类。参数由框架校验，`run()` 返回可 JSON 序列化的数据，失败时抛出异常，由注册表包装为错误结果。

```python
from pydantic import Field
from .tool import Tool, ToolArgs


class UppercaseArgs(ToolArgs):
    text: str = Field(min_length=1, description="Text to convert to uppercase.")


class UppercaseTool(Tool[UppercaseArgs]):
    args_schema = UppercaseArgs

    def __init__(self):
        super().__init__(
            name="uppercase", description="Convert text to uppercase.",
        )

    def run(self, parameters: UppercaseArgs) -> dict:
        return {"text": parameters.text.upper()}
```

保存后即可导入注册：

```python
from bpcagent import ToolRegistry
from bpcagent.tools.uppercase import UppercaseTool

registry = ToolRegistry()
registry.register_tool(UppercaseTool())
result = registry.execute_tool("uppercase", {"text": "hello"})
print(result.data)
```

已有同步函数也可通过 `registry.register_function(name, description, func, args_schema=Args)` 注册，函数的命名参数应与参数模型一致。需要提取器、数据库连接或保存目录时，通过工具构造函数注入；可参考 [bpc_patient.py](src/tools/bpc_patient.py) 用工厂统一创建依赖。

## 创建新智能体

参考 [PatientStateAgent](src/agents/patient_state.py)：在 `src/agents/` 新建文件，继承 `BasicAgent`，在构造函数中准备依赖、注册工具并固定提示词，通过 `run()` 接收任务。例如保存为 `src/agents/calculation.py`：

```python
from .agents import BasicAgent
from ..tools import CalculatorTool, ToolRegistry


class CalculationAgent(BasicAgent):
    def __init__(self, *, config=None, llm=None):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        super().__init__(
            name="CalculationAgent",
            config=config, llm=llm, tool_registry=registry,
            sys_prompt="Use the calculate tool for arithmetic. Respond in English.",
        )

    def run(self, input="Calculate 2 + 3 using the tool.", **kwargs):
        return super().run(input, **kwargs)
```

在应用中加载环境配置后调用：

```python
from dotenv import load_dotenv
from bpcagent import Config
from bpcagent.agents.calculation import CalculationAgent

load_dotenv(".env", override=False)
agent = CalculationAgent(config=Config(max_tool_iterations=6))
result = agent.run("Calculate 15 * 8 using the tool.")
print(result.output["text"] if result.finished else result.info)
```

普通智能体可直接复用父类工具循环。如果像 `PatientStateAgent` 一样要求“完成指定工具调用后立即停止”，还需参考它的 `_run_tool_loop()` 实现，在代码中检查完成条件，不能仅依靠提示词。

物理目录是 `src/agents`、`src/tools`，导入路径仍为 `bpcagent.agents`、`bpcagent.tools`。如需支持 `from bpcagent import NewAgent`，在对应子包的 `__init__.py` 和 `src/__init__.py` 中添加导出；工具同理。
