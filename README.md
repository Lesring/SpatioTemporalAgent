# 时空地理计算智能体（MultimediaExperiment）

依据时空地理计算基本算子，构建检索增强知识库，并利用大语言模型构建**分层智能体**，实现从自然语言到基本算子自动调用的时空地理计算。

## 先读这个

- **[`ABC对接指南.md`](./ABC对接指南.md)** — A/B/C 职责、最优框架、**锁定的 HTTP/JSON 契约（schema 1.0）**
- **[`spatio_temporal_agent/README.md`](./spatio_temporal_agent/README.md)** — 算子列表、CLI、评测、配置说明
- **[`examples/response_example.json`](./examples/response_example.json)** — 标准响应样例（给前端对照）

## 成员 B 已提供

| 能力 | 说明 |
|------|------|
| `TaskExecutor` | 带 id 的任务执行、异常兜底、统计 |
| 固定 `AgentResponse` | `answer` + `process.steps` + `meta` |
| HTTP API | `POST /v1/ask`、`POST /v1/ask/batch`（线程池） |
| 日志 | 统一 logging |

## 快速启动

```bash
pip install -r requirements.txt
cp spatio_temporal_agent/config_template.json spatio_temporal_agent/config.json
# 编辑 config.json 填入 LLM api_base / api_key / model

python -m spatio_temporal_agent.api.server
```

```bash
curl -X POST http://127.0.0.1:8000/v1/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"id\":\"demo\",\"query\":\"计算上海到北京的距离\"}"
```

## 目录概览

```
├── ABC对接指南.md              # 三方对接（必读）
├── README.md
├── requirements.txt
├── examples/response_example.json
└── spatio_temporal_agent/      # 主代码包（A 算子/知识库 + B Agent/API）
```

## 许可证

课程实验项目，仅供学习使用。
