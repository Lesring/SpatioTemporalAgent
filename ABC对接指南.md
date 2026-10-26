# ABC 对接指南（Agent 一看就会）

> **schema_version: `1.0`（已锁定）**  
> 变更接口必须三方确认并升版本号。  
> 本仓库由 **成员 B（Agent 核心层）** 维护编排与 HTTP 契约；A 提供算子/知识库/数据；C 做前端与评测。

---

## 0. 一句话分工

| 角色 | 负责什么 | 不要越界做什么 |
|------|----------|----------------|
| **A 基础资源层** | 地理算子实现与单测；知识条目整理与检索实现；业务数据（地名词典等）与说明文档；在线地理 API 适配（如 Nominatim） | 不改 B↔C 响应结构；不写前端 |
| **B Agent 核心层** | 架构与任务执行器；大模型 API / 工具调用；状态管理、异常、日志；**对外 HTTP 接口与固定 JSON 契约**；线程池批量执行 | 不随意改算子公式；不写评测前端页面 |
| **C 前端与评测层** | Web 页面；按契约调 B；测试集 / 指标 / 自动评测；图表报告演示 | 不绕过接口直接改 B 内部；不自造另一套响应字段 |

技术分层（与角色正交）：**算子层 → 知识库层 → 智能体层 → 数据层**。

---

## 1. 最优框架（B 已落地）

```
自然语言 query
    │
    ▼
TaskExecutor（B，唯一对外入口）
    │  1. KnowledgeBase 检索增强（A）
    │  2. LLM function calling 选算子（B）
    │  3. call_operator 执行公式（A）
    │  4. 组装固定 AgentResponse（B）
    ▼
HTTP JSON（schema 1.0）→ C 渲染 / 评测
```

**为何不用强绑 LangChain 等框架？**  
题目需要的是「分层智能体 + 工具调用闭环」。本项目用自研 `SpatioTemporalAgent` 循环 + OpenAI 兼容 tools，能力等价且依赖更轻。若后续要加多轮记忆/复杂工作流，可再评估 LangGraph，**不改变对外 schema**。

关键代码位置：

| 模块 | 路径 | 谁维护 |
|------|------|--------|
| 算子实现 | `spatio_temporal_agent/operators/` | A |
| 知识库 | `spatio_temporal_agent/knowledge_base/` + `data/knowledge_items.json` | A |
| 地名/地理数据 | `spatio_temporal_agent/geocoding.py` + `data/places.json` | A |
| LLM 客户端 | `spatio_temporal_agent/agent/llm_client.py` | B |
| 智能体编排 | `spatio_temporal_agent/agent/spatio_temporal_agent.py` | B |
| 任务执行器 | `spatio_temporal_agent/agent/executor.py` | B |
| **契约 schema** | `spatio_temporal_agent/agent/schema.py` | B（锁定） |
| **HTTP API** | `spatio_temporal_agent/api/server.py` | B |
| 评测脚本雏形 | `spatio_temporal_agent/evaluate.py` | C 可扩展 |
| Web 前端 | （C 新建，本仓库可不放） | C |

---

## 2. 环境与启动（B 服务）

```bash
pip install -r requirements.txt

# 配置 LLM（勿提交真实 Key）
cp spatio_temporal_agent/config_template.json spatio_temporal_agent/config.json
# 编辑 api_base / api_key / model

# 启动 API（默认 0.0.0.0:8000，线程池默认 4）
python -m spatio_temporal_agent.api.server
```

常用环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `AGENT_HOST` | 绑定地址 | `0.0.0.0` |
| `AGENT_PORT` | 端口 | `8000` |
| `AGENT_MAX_WORKERS` | 批量线程池大小 | `4` |
| `AGENT_MAX_BATCH_SIZE` | 单次批量上限 | `50` |
| `AGENT_CONFIG` | config.json 路径 | 包内 `config.json` |
| `AGENT_LOG_LEVEL` | 日志级别 | `INFO` |

健康检查：`GET http://127.0.0.1:8000/health`

---

## 3. B↔C 接口规则（锁定）

Base URL：`http://<host>:8000`  
所有业务响应含 `"schema_version": "1.0"`。

### 3.1 `POST /v1/ask` — 单条

**请求：**

```json
{
  "id": "case-001",
  "query": "计算上海到北京的距离"
}
```

**响应（固定结构）：**

```json
{
  "schema_version": "1.0",
  "id": "case-001",
  "success": true,
  "answer": "上海到北京的球面距离约为 1067.31 公里。",
  "process": {
    "kb_context": "...",
    "kb_hits": [
      {"title": "Haversine 距离公式", "description": "...", "keywords": ["距离", "..."]}
    ],
    "steps": [
      {
        "step": 1,
        "type": "kb_retrieve",
        "status": "ok",
        "name": "KnowledgeBase.search",
        "args": {},
        "result": {"preview": "..."},
        "message": "领域知识检索完成",
        "elapsed_ms": 0
      },
      {
        "step": 2,
        "type": "tool_call",
        "status": "ok",
        "name": "resolve_place",
        "args": {"name": "上海"},
        "result": {"success": true, "result": {"lat": 31.23, "lon": 121.47}},
        "message": "...",
        "elapsed_ms": 3.2
      },
      {
        "step": 3,
        "type": "tool_call",
        "status": "ok",
        "name": "haversine_distance",
        "args": {"lat1": 31.23, "lon1": 121.47, "lat2": 39.90, "lon2": 116.40},
        "result": {"success": true, "result": 1067310.12, "unit": "米"},
        "message": "...",
        "elapsed_ms": 0.4
      },
      {
        "step": 4,
        "type": "llm_answer",
        "status": "ok",
        "name": "final_answer",
        "args": {},
        "result": null,
        "message": "生成最终自然语言回答",
        "elapsed_ms": 0
      }
    ],
    "final_status": "completed"
  },
  "error": null,
  "meta": {
    "elapsed_ms": 1820.5,
    "model": "gpt-4o",
    "tool_call_count": 2,
    "operator_names": ["resolve_place", "haversine_distance"]
  }
}
```

**字段语义（C 必须按此解析）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 与请求一致，批量对账用 |
| `success` | bool | 是否拿到可用答案（`partial` 时仍可为 true） |
| `answer` | string | 给用户看的自然语言；失败时可能为空 |
| `process.final_status` | string | `completed` \| `partial` \| `failed` \| `pending` |
| `process.steps[]` | array | **有序**过程；`type` ∈ `kb_retrieve` \| `tool_call` \| `llm_answer` \| `error` |
| `process.steps[].status` | string | `ok` \| `error` \| `skipped` |
| `error` | string\|null | 失败原因；成功时为 `null` |
| `meta.elapsed_ms` | number | 端到端耗时 |
| `meta.operator_names` | string[] | 实际调用的算子名（评测算子命中率用） |

失败时**结构不变**，`success=false`，`process.final_status="failed"`，`error` 有文案。

### 3.2 `POST /v1/ask/batch` — 批量（多线程）

**请求：**

```json
{
  "items": [
    {"id": "1", "query": "计算上海到杭州的距离"},
    {"id": "2", "query": "判断点(31.25,121.45)是否在多边形内..."}
  ]
}
```

规则：

- 每条必须有唯一 `id`
- 服务端用线程池并发；默认 `AGENT_MAX_WORKERS=4`（LLM IO 密集，过大易触发限流）
- 单次 `items` 上限默认 50
- **响应 `results` 顺序与请求 `items` 顺序一致**（同时每条自带 `id`）

**响应：**

```json
{
  "schema_version": "1.0",
  "count": 2,
  "results": [ { "...单条 AgentResponse..." }, { "..." } ]
}
```

### 3.3 辅助端点

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/health` | 探活、模型名、算子列表、stats |
| GET | `/v1/operators` | 算子名 + function schema（C 可做说明页） |
| GET | `/v1/stats` | 累计 success/failed |

---

## 4. A 对接约定

1. **新增算子**：在 `operators/geo_operators.py` 实现 → `operators/registry.py` 注册 schema → 保证 `call_operator` 返回统一 dict：`success` / `result` / `explanation`（失败含 `error`）。
2. **知识库**：扩 `data/knowledge_items.json`（`title` / `description` / `keywords`）；检索逻辑在 `knowledge_base/`。升级向量 RAG 时保持 `search` / `build_context` 接口。
3. **数据层**：`places.json` 作本地缓存；未知地名走在线 API 并写回。不要追求「存下全世界 POI」；「附近 2km 公园」类需另提供 POI 检索算子（Overpass 等），由 A 实现、B 自动可调。
4. **禁止**：改 `agent/schema.py` 字段名或删减 `process.steps` 结构。

---

## 5. C 对接约定

1. 前端只依赖 **§3 契约**；逐步展示用 `process.steps`。
2. 评测建议指标（对齐题目）：
   - **算子选择准确率**：期望算子 ⊆ `meta.operator_names`
   - **业务流程准确率**：`success && final_status in (completed, partial)` 且算子命中
   - **地名解析率**：涉及地名的用例是否调用且成功 `resolve_place`
   - **知识库完整度**：条目覆盖率 + 检索命中率（可另调 A 的检索接口或看 `kb_hits`）
3. 测试用例 **由 C（或全组）自建**；仓库 `evaluate.py` 仅是 8 条样例。
4. 「AI 能聊两句」**不算**业务流程准确；必须以工具调用链为准。
5. 大量语料务必走 `/v1/ask/batch`，用 `id` 对账；注意限流，可把 `AGENT_MAX_WORKERS` 调到 2~4。

示例前端伪代码：

```javascript
const res = await fetch("http://127.0.0.1:8000/v1/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ id: "ui-1", query: userText }),
});
const data = await res.json();
// data.answer → 主回答区
// data.process.steps → 过程时间线
// data.meta → 调试/评测
```

---

## 6. 评测与题目指标（共识）

| 指标 | 怎么评 | 谁主责 |
|------|--------|--------|
| 领域知识库完整度 | 条目是否覆盖主算子/场景；检索是否命中 | A 内容，C 指标与报表 |
| 业务流程准确率 | 固定测试集上：选对算子 + 执行成功 + 有答案 | C 用例与跑分，B 保证可观测字段 |
| 地理计算准确率 | 算子公式可复核（通常弱化） | A |

「检索增强」= 先检索知识再注入 LLM 上下文（当前为关键词 RAG，可升级向量）。

---

## 7. 联调检查清单

- [ ] A：离线 `python -m spatio_temporal_agent.demo_offline` 算子与知识库 OK  
- [ ] B：配置 `config.json` 后 `python -m spatio_temporal_agent.api.server`，`/health` 返回 ok  
- [ ] B↔C：单条 `/v1/ask` 字段齐全；批量 id 顺序正确  
- [ ] C：前端按 `steps` 渲染过程；评测脚本只读契约字段  
- [ ] 密钥：`config.json` 已 gitignore，勿提交真实 API Key  



**当前锁定版本：`1.0`。**
