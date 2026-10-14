# Task 封装

`src/tasks/base.py` 同时定义抽象 `Task` 和 `TaskResult`。基类只规定 `run(...) -> TaskResult`，不规定配置、工作目录、Agent 类型或执行步骤；具体子类定义自己的查询参数。Task 不继承 Agent。

`PatientStateTask` 封装“提取 state/evidence → Docker sandbox 单位处理 → 收集产物”的现有流程。每次调用创建新的 `UnitPatientStateAgent`；不改动 `PatientStateAgent`，也不复制工具循环、转换规则或验收逻辑。

## 初始化一次，每次只传查询变量

以下示例在 `BPCAgent` 仓库目录运行。模型配置由应用加载；Task 不查找 `.env`，也不从包安装位置推测数据路径。

```python
from pathlib import Path
from dotenv import load_dotenv
from bpcagent import Config, PatientStateTask, PatientStateTaskConfig, SandboxConfig

load_dotenv(".env", override=False)
root = Path.cwd().parent
task = PatientStateTask(PatientStateTaskConfig(
    data_dir=root / "Pathway-Agent" / "dataset",
    rules_path=root / "Pathway-Agent" / "config" / "field_rules.json",
    workspace_root=root / "workspace",
    agent_config=Config(max_tool_iterations=10),
    sandbox_config=SandboxConfig(image="bpcagent-sandbox:0.1"),
))

result = task.run(patient_id="GENIE-DFCI-051542", cutoff_day=220)
if result.finished:
    print(result.agent_result.output["text"])
    print(result.artifacts["normalized"])
else:
    print(result.info)
print(result.task_dir)
print(result.log_path)
```

Docker 和镜像需事先准备，详见 [sandbox.md](sandbox.md)。Task 始终传入有效的 `SandboxConfig`，不提供自动回退到宿主机执行的选项。

`PatientStateTaskConfig` 字段：

| 字段 | 约定 |
| --- | --- |
| `data_dir` | 必填，已存在的数据集目录 |
| `workspace_root` | 必填，任务总目录；可尚未创建 |
| `rules_path` | 可选；提供时必须存在，省略时沿用提取器默认规则 |
| `agent_config` | 复用 `Config`，默认每阶段最多 10 轮 |
| `sandbox_config` | 复用 `SandboxConfig`，默认镜像 `bpcagent-sandbox:0.1` |
| `timezone` | 目录时间的 IANA 时区，默认 `Asia/Shanghai` |

相对路径在配置对象创建时转换成绝对路径，切换当前目录不会改变数据或工作区位置。Task 初始化时通过 `Config.resolve()` 保存模型配置快照，后续运行不重新采用变化的环境配置。构造 Task 不请求模型、不创建任务目录。

`run()` 仅接受关键字参数 `patient_id`、`cutoff_day` 和可选的 `cancer_id`。`cutoff_day` 是现有接口的相对天数，允许负数；不接受布尔值、浮点数或日期字符串。患者 ID 原样用于查询，不自动清洗或替换；空值、路径分隔符、控制字符等不安全目录字符会被拒绝。

## 目录和结果

```text
workspace/YYYYMMDD_HHMMSS_patient-id_cutoff-day/
  input/
  scripts/
  output/
  tmp/
```

同秒、同患者、同截止日的重复运行会抛出 `FileExistsError`，不复用或覆盖已有目录。等待下一秒后可再提交同一查询。每次运行使用独立 Agent，因此患者绑定、对话和工具状态不会跨任务沿用。

`TaskResult` 字段：

| 字段 | 含义 |
| --- | --- |
| `finished` | 只有 Agent 明确成功且必要产物存在时才为 `True` |
| `task_dir` | 本次任务的绝对路径 |
| `artifacts` | 已存在的文件路径，键为 `state`、`evidence`、`normalized`、`script` |
| `log_path` | 已存在的 Agent 日志绝对路径；不可用时为 `None` |
| `agent_result` | 原始 `AgentResult`，保留报告、轨迹和 Agent 完成状态 |
| `info` | Agent 信息的副本及 Task 层诊断 |

失败时仍收集已生成的文件，但文件存在不代表任务完成。提取文件有多个版本时按数值序号选择最新版本；只收集当前截止日的文件，跳过符号链接和工作区外路径。不从模型的文字报告中解析文件位置，也不重新执行生成的脚本。

如果 Agent 已成功而必要文件缺失，Task 返回 `finished=False`、`info.stop_reason=incomplete_artifacts`，保留原始 AgentResult 不变。配置、目录重名和程序异常继续抛出，不伪装成正常结果。打印和退出码由调用方处理。

任务不另存结果 JSON，不生成 `run_result.json`。沿用 Agent 的 `log/UnitPatientStateAgent_*.json`，日志目录仍相对于应用执行时的当前目录；Task 将它转换为绝对路径返回。

第一版不包含任务队列、批量调度、评分/奖励、断点恢复或通用 Agent 工厂。未来替换/合并 Agent 时，优先修改 `PatientStateTask._create_agent()`，保持查询接口和结果结构稳定。
