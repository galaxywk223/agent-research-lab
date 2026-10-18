# Agent Research Lab

个人打造的多智能体深度研究工作台：把研究编排、业务持久化、失败恢复与协议化评估放在同一套本地可跑的系统里。

## 能力

- **研究编排**：基于 LangGraph 的规划、检索、综合与报告生成
- **业务持久化**：PostgreSQL 记录 Research / Task / Attempt / Evidence，与运行时 checkpoint 解耦
- **可靠性**：Attempt lease + heartbeat；过期 Attempt 进入 `stale_lease_expired`，支持至少一次语义下的 stale recovery
- **评估与可观测**：冻结评测协议、题库与 rubric；支持 fake provider 做 API E2E
- **工作台 UI**：Next.js 研究对话界面

## 仓库结构

```text
backend/src/agent_research_lab/  # LangGraph agent、持久化、metrics
frontend/                        # Next.js 工作台
eval/                            # 评测协议、题库、runner
migrations/                      # Alembic 迁移
scripts/                         # 本地启动脚本
tests/                           # pytest
```

## Quickstart

### 1. 环境

复制 `.env.example` 为 `.env`，填入模型与搜索相关密钥，并配置 Postgres：

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_PORT
DATABASE_URI
AGENTLAB_DATABASE_URI
```

### 2. 启动数据库并迁移

```powershell
docker compose up -d postgres
uv run alembic upgrade head
```

### 3. 启动后端与前端

一键本地启动：

```powershell
.\scripts\start-local.ps1
```

或分别启动：

```powershell
uv run --project . langgraph dev --no-browser --no-reload --port 2024
```

```powershell
cd frontend
pnpm install
pnpm dev
```

### 4. Fake provider（无真实模型/搜索调用）

```powershell
$env:AGENTLAB_PROVIDER_MODE = "fake"
```

### 5. 测试

```powershell
uv run pytest -q
```

未配置 `DATABASE_URI` 时，需要数据库的集成测试会自动跳过。

## 设计要点

- `thread_id` 是对话标识，`research_id` 是逻辑研究标识，`run_id` 是一次 LangGraph 执行
- Evidence 在 Session 内按规范化 URL 去重，并通过 `task_attempt_evidence` 被多个 Attempt 引用
- continuation 需携带 checkpoint 上下文与 state 中的 `research_id`；缺失时显式失败
- LangGraph checkpoint 仍由本地 `.langgraph_api` runtime 管理；Postgres 负责业务审计与 Attempt 生命周期

## License

MIT © Wang Kai
