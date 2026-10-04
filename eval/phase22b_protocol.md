# Phase 2.2B Development Protocol

## 数据身份与停止点

- dataset_split：`development`。
- purpose：`strategy_development`。
- dev_version：`dev-v1`。
- seed：`20261004`。
- 五题题集：`dev_questions_v1.jsonl`，D01—D05。
- Development SHA-256：`bdd48cb7cc831d1b973647e816c541087ae53d5c3f26542f2daada936dd0dcd9`。
- Development manifest：`dev_manifest_v1.json`。
- Holdout SHA-256：`9511f18eb766976a115d35768f56ee6910172458e69507082a42da867883bf76`。

题集覆盖生态保护、平台产业、材料工程、巴西支付基础设施、工业史。题目均限定历史范围且使用公开资料；requirements 不包含标准答案。Development loader 不解析 holdout 问题，真实 holdout Research/Judge 请求均被拒绝。冻结文件按原始 UTF-8/LF 字节校验，后续题目修改需要新版本。

本阶段仅执行五题首次真实 baseline 及 completed 题的 Coverage/Quality。Adaptive Gate 仅形成设计；holdout 与 adaptive execution 保持暂停。

## 两个配置快照

|快照|文件|SHA-256|
|---|---|---|
|Baseline|`configs/baseline_v1.json`|`77800b109e501da893f828a698b651e829cb23a72db2c5ac25863f6e45d7ef28`|
|Judge|`configs/judge_phase2_v1.json`|`f20076564ce3f7880b3d54ac438041d002d285247c02129862a61d4a400ef9cf`|

Baseline 四个模型均为 `openai:qwen3.7-flash`，Tavily 搜索。并发 5、Supervisor 配置 `max_researcher_iterations=6` 及现有 `>` 边界、Researcher tool rounds 10。clarification、provider retry、token/content limits、Prompt、Task decomposition、Evidence persistence 保持原行为。运行结果必须包含并匹配快照中的每个字段；缺失字段不默认有效。

Judge 模型为 `openai:glm-5.3`，method 为 `json_schema`，reasoning_effort 为 `low`。请求 temperature 为 0，实际生效状态为 `unknown`；max_tokens 为 8192，max_retries 为 0。调用使用 Chat Completions，`use_responses_api=False`，不发送 `enable_thinking=false`。输入显式包含 `output_json_schema`，输出使用 `include_raw=True`。schema、Prompt、rubric 和依赖版本均由 Judge 快照校验。

## 运行 journal 与恢复

Thread ID 和 request identity 在网络操作前落盘。每次显式 Research retry 使用独立 Thread、attempt_no 和历史结果文件。Run metadata 包含 experiment、split、question、attempt 和 request identity。

- 未知 Run 创建结果：查询原 Thread，禁止自动重发。
- running/timeout：挂接原 Run，禁止替代执行。
- Research 成功但收集失败：只重新收集 metrics/Evidence。
- 需要澄清或 runtime interruption：保存 incomplete，禁止自动生成用户回答。
- failed：默认保留；只有 `--resume --retry-failed` 可以创建下一次 Research。
- Judge：Coverage/Quality 独立 journal；成功阶段复用，失败默认不重试。中断的付费 Judge 调用结果未知时不自动重发。
- 恢复前：校验 split、题集、Baseline、Judge、Prompt、rubric 和执行源码身份。

Research API、业务、收集状态分别保存。API success 不等价于业务 completed。网络 timeout 只限制 Runner 等待时间，原 Run 继续存在；后续观察不得覆盖其历史身份。

## 运行隔离

Development 独立 API 工作目录为 `.runtime/dev-api/`，checkpoint 由该目录内 `.langgraph_api` 管理。独立 PostgreSQL 数据库为 `agentlab_dev_v1`；测试数据库为 `agentlab_dev_tests`。数据库沿用既有 Alembic migrations，无新业务 schema。metrics 使用绝对目录 `.runtime/metrics-dev-real`。

D01 初次启动使用历史 acceptance 数据库及工作区 checkpoint 目录，存在启动隔离偏差。原 Run 保留且不重发；终态业务记录复制到独立 Development 数据库，保持原 research/session/task/attempt/evidence 身份。D02—D05 的首次执行使用独立 API 和数据库。结果 manifest 与 D01 journal 明确保存该偏差；该操作不将 D01 描述为最初即完全隔离。

## 离线分析语义

### 成本

七个角色分别报告 calls、observed input/output/total tokens、累计和平均调用 latency。usage 缺失保持 null，统计包含 mean、median、min、max、有效 n 和 missing count。Research wall-clock 与并行调用累计耗时分别报告。retry_usage_included 为 `unknown`，observed tokens 不称为绝对 billing tokens；Research 和 Judge 成本分别统计，货币费用未知。

### Summarization fan-out

逐题报告 tool/query/provider request、raw/deduped result、summarization calls/tokens 和数据库 Evidence。四个 ratio 为 calls/query、calls/provider request、calls/deduped result、summarization tokens/Evidence；零或未知分母返回 null。

现有 LLM event 没有 Task/query association ID，摘要 tokens 不能严格分配到某个 query/result。Task 只报告已有可证明总量，不补造按角色的分摊。

### Round 与 low gain

每个 Task 保留完整 round 序列。所有轮与搜索轮分别统计；每个 round index 显示有效 n。后期统计仅包含继续执行的 Task，存在 survivorship。

零新增 URL、零新增 domain、连续零新增搜索轮及重复率原始案例仅标 `candidate_low_gain`。无搜索轮不作为检索耗尽。重复率 0.8 仅用于描述性案例展示，不是选定 Gate threshold。完整搜索轮分布和排序独立保存。

### Task overlap

topic 使用 Unicode NFKC、casefold 和空白标准化，tokens 由 Latin 词和中文双字片段构成。Task pair 分别计算 topic、URL、domain Jaccard。空集合或缺失集合返回 null，并保存状态和集合大小；pair n=0 不表示无重叠。不使用 embedding。

### Evidence 利用 proxy

数据库 Evidence URL/domain 集合与 final_report 正文实际出现的 URL/domain 集合分别提取。Runner 的全部 source_urls 不充当已引用来源。可识别的 `append_sources` 末尾列表保守排除；raw model report 未保存时，自动追加的精确 provenance 保持 unsupported。该 proxy 不验证 claim support，也不能证明某条 Evidence 被阅读或使用。

### 质量与 Gate 设计

Coverage labels 及五维 Quality 逐题保存。Evidence context 为截断的 `retrieved_research_notes`，评分反映报告与这些笔记的一致性，不构成网页原始事实核验。

候选信号只能提供设计依据。threshold/patience 未获得充分支持时保持未确定；最终报告质量不能证明某个前缀能够安全停止。offline Coverage/Evidence sufficiency 不属于免费在线信号。

hard budget 保留现有并发、Supervisor 与 Researcher 上限。Baseline 没有全局 LLM call/Evidence 上限，设计中这些字段为未定义。未来 10×2 paired 情景只缩放已观察的 baseline 数据；adaptive 实际成本和节省比例未知。Judge 单列 50 次计划调用：Coverage 20、Quality 20、Pairwise 10。

## 验收资产

- `tests/test_development.py`：split/hash、恢复、创建断线、timeout、历史 attempt、Judge wire contract 和独立阶段恢复。
- `tests/test_characterization.py`：缺失 usage、fan-out ratio、round n、搜索轮过滤、overlap、正文 URL proxy、混合身份拒绝和成本缩放。
- fake 与 real 结果使用独立 experiment 目录，禁止混合模式或 split。
- PostgreSQL 并发/lease 测试实际执行，数据库 skip 不计作验收通过。

## 本轮实际限制

真实五题首次执行形成 2 个 completed、3 个 failed。网关在 D03 中途及 D04/D05 首次请求返回 `AccessDenied.Unpurchased`；该响应不能证明具体账户或计费原因。失败 Run 未自动重试，模型与预算未切换。

D03—D05 的 fatal exception 未触发既有 instrumentation finalize，metrics 文件缺失；API 状态为 error，而数据库 Session/session_run 仍保留 running。结果记录这些差异，不伪造失败成本、不人工覆盖业务终态。已完成答案仅有两个，有完整 Research token 总量的样本仅有一个；Gate threshold/patience 未确定。
