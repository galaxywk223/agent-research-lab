# Synthetic Judge Acceptance

## Scope

验收仅包含 `openai:deepseek-v4-pro-0813` 和 `openai:glm-5.3`。输入来自虚构配送案例，不使用 Development 或 Evaluation 问题。现有 Research 策略、evaluator、环境配置和冻结资产保持不变。

## Execution

项目 `uv` 环境提供 LangChain、OpenAI SDK 和 Pydantic。凭据从本地 `EVAL_JUDGE_API_KEY` / `EVAL_JUDGE_BASE_URL` 读取；缺失时使用本地 OpenAI-compatible 配置。认证 header 不进入 journal，错误内容按凭据值脱敏。

```powershell
uv run pytest -q tests/test_judge_acceptance.py
uv run python -m eval.judge_acceptance --execute-real --output-dir eval/runs/judge-acceptance-v1
```

真实验收最多计划 136 次逻辑调用，顺序执行，客户端 `max_retries=0`，单次发送 `max_tokens=8192`。两候选均先验证基础生成，再执行参数探测和 structured matrix；无基础权限或无可靠主方法时，后续项目标记 `not_run`。

## Artifacts

输出目录包含：

- `manifest.json`：候选、依赖版本、fixture/schema/prompt/rubric hash、受保护文件 hash及调用预算。
- `synthetic_inputs.json`：三个固定案例、接近质量的改写答案和预期方向。
- `inputs/`：约 20,000 字符笔记及 20,000 字符报告的长度验收输入。
- `calls/`：每次请求的独立 atomic journal、脱敏请求机制与 provider response、解析和校验结果。
- `summary.json`：全部尝试的分母、评分波动、方向检查、位置差异、usage和latency。
- `recommended_config.json`：验收后的配置建议，不自动写入正式环境。

再次运行同一命令复用已结束的验收调用，包括失败调用。`starting` journal 表示请求结果不明；工具显式停止，避免重复发送造成隐含重试或重复费用。

## Interpretation

`json_schema`、`function_calling` 和 `json_mode` 分别显式发送。实际 HTTP `response_format` / tool schema 用于检测客户端 fallback。正式建议只选择三种 schema 均成功的 `json_schema` 或 `function_calling`；`json_mode` 仅用于诊断。

Pydantic schema 与现有 evaluator 一致。Coverage 额外校验四个 requirement 均被准确评分一次。截断、字段缺失、非法 enum、越界数字和 fallback 不计为成功。

标准差采用本次有效重复观测的 population standard deviation。评分错误不填零，API/parse/validation/error rate 保留全部尝试的分母。接近质量对的 tie rate 与明显优劣对的逻辑胜率分别报告。

参数被 HTTP 接受不证明参数生效。`temperature_effective` 和 `thinking_mode_confirmed` 在缺少可靠服务端证据时保持 `unknown`；reasoning 内容与 reasoning token details 单独记录。重复输出一致不能证明绝对确定性。

Token 数只使用 provider 返回的 usage。累计 latency 是顺序验收调用耗时，不是 Research benchmark 成绩。实际计费、未暴露的 provider 内部重试及 reasoning 计费语义仍可能无法完整确认。

## Stop Condition

验收结束后仅给出两个候选的实测比较及固定调用配置建议。Development Run、Evaluation Run 和 Adaptive Gate 不在该工具执行范围内。
