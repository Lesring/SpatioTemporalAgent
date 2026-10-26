# 时空地理计算智能体（Spatio-Temporal Geospatial Agent）

依据时空地理计算基本算子，构建检索增强知识库（RAG），调用 OpenAI 兼容 LLM，实现**自然语言输入 → 算子自动调用 → 结果输出**的完整分层智能体流程。

> **ABC 分工与接口契约见仓库根目录 [`ABC对接指南.md`](../ABC对接指南.md)（schema `1.0` 已锁定）。**  
> 本包中 **B 层**提供 `TaskExecutor` + HTTP API（`/v1/ask`、`/v1/ask/batch`）。

核心功能：
- **16 个基本算子**（空间 12 + 时间 4，支持 LLM function calling）
- **地名自动解析**：自然语言提及地名时自动查本地词典 → 联网 Nominatim 兜底 → **写回数据库**
- **多步任务编排**：智能体可串联多个算子完成复杂查询
- **固定结构过程响应**：`answer` + `process.steps` + `meta`（供前端与评测）
- **HTTP + 线程池批量**：带 `id` 的单条/批量接口
- **量化评测脚本**：输出算子选择准确率、地名解析准确率、整体流程成功率

## 1. 项目结构

```
spatio_temporal_agent/
├── __init__.py
├── __main__.py              # python -m 入口
├── main.py                  # 交互式 CLI 入口
├── demo_offline.py          # 离线演示（无需 LLM，验证算子+知识库+容错）
├── test_llm.py              # 端到端 LLM 测试（6 个场景，含地名/时间算子）
├── evaluate.py              # 量化评测（8 用例，输出三类准确率指标）
├── geocoding.py             # 地名解析器（本地词典 + Nominatim 在线兜底 + 写回 places.json）
├── config.json              # LLM 配置（已填入 Agnes-AI API Key）
├── config_template.json     # 配置模板（新建配置时使用）
├── README.md
├── 中期检查演示脚本.md
│
├── operators/               # 【算子层】16 个基本算子
│   ├── __init__.py          # 导出所有算子
│   ├── geo_operators.py     # 算子实现（空间 + 时间，纯 Python）
│   └── registry.py          # 算子注册表 + OpenAI function schema + call_operator
│
├── knowledge_base/          # 【知识库层】领域知识 RAG
│   └── __init__.py          # KnowledgeBase（关键词检索，可升级为向量 RAG）
│
├── agent/                   # 【B 智能体层】LLM 编排
│   ├── __init__.py
│   ├── llm_client.py        # OpenAI 兼容客户端（httpx，429/网络异常自动重试）
│   ├── spatio_temporal_agent.py  # 智能体主循环（工具调用 + 多步编排）
│   ├── executor.py          # TaskExecutor：对外唯一执行入口
│   ├── schema.py            # B↔C 锁定响应结构 schema 1.0
│   └── logging_utils.py     # 统一日志
│
├── api/                     # 【B HTTP】供 C 对接
│   ├── __init__.py
│   └── server.py            # FastAPI：/v1/ask、/v1/ask/batch
│
└── data/                    # 【数据层】
    ├── knowledge_items.json # 领域知识条目（8 条）
    └── places.json          # 地名词典（Nominatim 可自动写回扩充）
```

## 2. 支持的 16 个基本算子

### 空间算子（12 个）

| 算子名 | 功能 | 主要参数 |
|--------|------|----------|
| `haversine_distance` | 两点球面距离（米） | 两个点的经纬度 |
| `bearing` | 起点到终点的方位角（度） | 两个点的经纬度 |
| `destination_point` | 起点 + 距离 + 方位角 → 终点坐标 | 起点、距离、方位角 |
| `point_in_polygon` | 判断点是否在多边形内（射线法） | 点、多边形顶点 |
| `polygon_area` | 多边形近似面积（平方米） | 多边形顶点 |
| `bounding_box_overlap` | 两个边界框是否重叠 | 两个 BBox（各 4 个数值） |
| `buffer_circle` | 圆形缓冲区边界框 | 圆心、半径 |
| `nearest_point` | 找最近点（多候选点中距离目标最近者） | 目标点、候选点列表 |
| `polygon_centroid` | 多边形质心坐标 | 多边形顶点 |
| `line_length` | 折线/线段总长度（米） | 折线顶点列表 |
| `angle_between` | 两个方位角之差（度） | 两个角度 |
| `resolve_place` | **地名 → 坐标解析**（见下方说明） | 地名（自然语言） |

### 时间算子（4 个）

| 算子名 | 功能 | 主要参数 |
|--------|------|----------|
| `temporal_distance` | 两个时间戳之间的时间差（秒/分钟/小时） | 两个 ISO 8601 时间戳 |
| `timezone_convert` | 时区转换（如 UTC → 北京） | 时间戳、源时区、目标时区 |
| `time_range_filter` | 过滤时间范围内的事件 | 事件时间列表、起止时间 |
| `time_series_resample` | 时间序列重采样（按时间间隔取均值） | 数据点列表、重采样间隔 |

### `resolve_place` 算子说明（地名自动解析）

**调用时机**：当用户自然语言中提及地名（如"上海"、"北京"）而非直接给经纬度时，智能体先调用 `resolve_place` 将地名转为坐标，再调用空间算子。

**三层解析逻辑**（`geocoding.py`）：

```
resolve_place(name)
  1. 查本地 data/places.json 词典（含中文别名/英文/拼音）
       命中 → 直接返回坐标（毫秒级）
  2. 本地未命中 → 联网调用 Nominatim（OpenStreetMap 免费接口）
       命中 → 返回坐标 + 自动写回 places.json（下次直接离线命中）
  3. 全部未命中 → 返回失败信息，LLM 告知用户
```

当前词典已收录 55+ 地名（北京、上海、广州、黄山、故宫、西湖、外滩等），并含中英文/拼音别名。

## 3. 使用方式

### 3.1 安装依赖

```bash
pip install -r requirements.txt   # 仅需 httpx
```

### 3.2 配置 LLM（交互式运行需要）

```bash
# 方式一：复制模板并填入 Key
cp config_template.json config.json
# 编辑 config.json，填入 api_base / api_key / model

# 方式二：环境变量
export LLM_API_BASE=...
export LLM_API_KEY=...
export LLM_MODEL=...
```

不同服务商 `api_base` 示例：
- OpenAI：`https://api.openai.com/v1`，model `gpt-4o`
- DeepSeek：`https://api.deepseek.com/v1`，model `deepseek-chat`
- 阿里百炼/通义：`https://dashscope.aliyuncs.com/compatible-mode/v1`

### 3.3 离线演示（无需 LLM，推荐先跑）

```bash
python -m spatio_temporal_agent.demo_offline
```

演示三块内容：
1. 算子直接调用（验证计算正确性）
2. 知识库检索（验证领域知识命中）
3. 错漏案例（参数错误 / 未知算子 的容错）

### 3.4 端到端 LLM 测试（6 个场景，含地名自动解析与时间算子）

```bash
python -m spatio_temporal_agent.test_llm
```

### 3.5 量化评测（输出三类准确率指标）

```bash
python -m spatio_temporal_agent.evaluate
```

- **算子选择准确率**：LLM 是否正确选到目标算子
- **地名解析准确率**：涉及地名的用例中 `resolve_place` 是否成功
- **整体流程成功率**：端到端任务全部完成的用例占比

### 3.6 启动 HTTP API（B↔C 对接）

```bash
python -m spatio_temporal_agent.api.server
# 默认 http://0.0.0.0:8000
# 可选：set AGENT_MAX_WORKERS=4
```

```bash
# 单条
curl -X POST http://127.0.0.1:8000/v1/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"id\":\"1\",\"query\":\"计算上海到北京的距离\"}"

# 批量（多线程，按 id 回填）
curl -X POST http://127.0.0.1:8000/v1/ask/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"items\":[{\"id\":\"a\",\"query\":\"...\"},{\"id\":\"b\",\"query\":\"...\"}]}"
```

完整字段说明见根目录 `ABC对接指南.md`；样例 JSON 见 `examples/response_example.json`。

### 3.7 交互式运行

```bash
python -m spatio_temporal_agent
# 或
python spatio_temporal_agent/main.py
```

输入自然语言地理/时间计算问题，例如：
- "计算上海和北京的距离"（自动解析地名，无需手输经纬度）
- "判断点 (31.25, 121.45) 是否在多边形 [[31.2,121.4],[31.2,121.5],[31.3,121.5],[31.3,121.4]] 内部"
- "以 (31.23,121.47) 为圆心生成 5000 米缓冲区"
- "上海到黄山的方位角"
- "2024-03-01T08:00:00Z 到 2024-03-01T18:30:00+08:00 相隔多久"

## 4. 智能体执行流程

```
用户自然语言
   │
   ▼
检索领域知识库（KnowledgeBase.search）
   │
   ▼
构建 system prompt（含知识库上下文 + 16 个算子 schema）
   │
   ▼
调用 LLM（OpenAI tools / function calling）
   │
   │  system prompt 指示：
   │    - 地名（如"上海"）先调 resolve_place 取坐标，再计算
   │    - 时间任务（时区转换/时间差）调 temporal_* 算子
   │    - 复杂任务可多步串联（resolve_place → haversine_distance → ...）
   │
   ├─ 模型选择算子 + 提取参数 ──▶ call_operator(name, **args)
   │        ▲                              │
   │        └──── 结果回填 tool message ◀──┘
   ▼
LLM 组织最终自然语言回答
   │
   ▼
输出：answer + 调用记录 + 知识库上下文
```

### 地名自动解析与自更新数据库

当 LLM 调用 `resolve_place(name)` 时，`geocoding.py` 中的 `Geocoder` 执行三层解析：

1. **本地词典**（`data/places.json`，55+ 地名）：命中即返回，毫秒级
2. **Nominatim 在线兜底**（OpenStreetMap 免费接口，无需 Key）：本地未命中时联网查询
3. **写回数据库**：在线命中后自动追加到 `places.json`，下次同类地名直接离线命中

因此用户提问时无需手动输入经纬度，自然语言提及地名即可；词典随使用量持续扩充。

## 5. 评测指标对齐

- **领域知识库完整度**：见 `data/knowledge_items.json`（领域知识条目）和 `data/places.json`（地名词典），可继续扩充。
- **业务流程准确率**：`evaluate.py` 输出算子选择准确率、地名解析准确率、整体流程成功率三类量化指标。
- **地理计算准确率**：题目明确不要求，算子采用标准公式（Haversine / 射线法 / 鞋带公式），结果可复核。

## 6. 后续可扩展

- 将 `KnowledgeBase` 从关键词匹配升级为向量检索（embedding + 向量库）。
- 扩充 `knowledge_items.json` 领域知识条目。
- 增加更多算子（路径规划、时空叠加等）。
- 增加多轮对话记忆、参数缺失自动追问。
