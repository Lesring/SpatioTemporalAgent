"""FastAPI 服务：B↔C 锁定接口。

端点：
- GET  /health              健康检查 + 线程池状态
- GET  /v1/operators        可用算子列表（C 可展示）
- POST /v1/ask              单条：{id, query} → AgentResponse
- POST /v1/ask/batch        批量：{items:[{id,query},...]} → {results:[...]}
- GET  /v1/stats            执行器累计统计

批量请求走 ThreadPoolExecutor，响应按 id 回填，结构固定。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..agent.executor import TaskExecutor
from ..agent.llm_client import LLMClient
from ..agent.logging_utils import get_logger, setup_logging
from ..agent.schema import SCHEMA_VERSION, empty_failed_response
from ..operators.registry import get_all_schemas, list_operator_names

logger = get_logger()

# 默认线程池大小：LLM 多为 IO 等待，4~8 较稳；可用环境变量覆盖
_DEFAULT_MAX_WORKERS = int(os.environ.get("AGENT_MAX_WORKERS", "4"))
_MAX_BATCH_SIZE = int(os.environ.get("AGENT_MAX_BATCH_SIZE", "50"))


class AskRequest(BaseModel):
    id: str = Field(..., description="请求唯一 id，响应原样带回")
    query: str = Field(..., description="自然语言任务")


class BatchAskRequest(BaseModel):
    items: list[AskRequest] = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str
    schema_version: str
    model: str
    max_workers: int
    operators: list[str]
    stats: dict[str, int]


def _resolve_config_path() -> Path:
    env = os.environ.get("AGENT_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config.json"


def create_app(
    executor: Optional[TaskExecutor] = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> FastAPI:
    setup_logging(os.environ.get("AGENT_LOG_LEVEL", "INFO"))

    app = FastAPI(
        title="Spatio-Temporal Agent API",
        description="成员 B：Agent 核心层 HTTP 接口（schema_version=" + SCHEMA_VERSION + "）",
        version=SCHEMA_VERSION,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {
        "executor": executor,
        "pool": None,
        "max_workers": max(1, max_workers),
        "model": "",
    }

    @app.on_event("startup")
    def _startup() -> None:
        if state["executor"] is None:
            cfg = _resolve_config_path()
            try:
                llm = LLMClient.from_config_file(str(cfg))
            except ValueError as e:
                logger.error("LLM 配置失败: %s", e)
                raise RuntimeError(str(e)) from e
            state["executor"] = TaskExecutor(llm=llm)
            state["model"] = llm.model
            logger.info("TaskExecutor ready model=%s config=%s", llm.model, cfg)
        else:
            state["model"] = getattr(state["executor"].llm, "model", "")
        state["pool"] = ThreadPoolExecutor(
            max_workers=state["max_workers"],
            thread_name_prefix="agent-worker",
        )
        logger.info("thread pool started max_workers=%s", state["max_workers"])

    @app.on_event("shutdown")
    def _shutdown() -> None:
        pool = state.get("pool")
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _exec() -> TaskExecutor:
        ex = state.get("executor")
        if ex is None:
            raise HTTPException(status_code=503, detail="executor not ready")
        return ex

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ex = state.get("executor")
        return HealthResponse(
            status="ok" if ex is not None else "starting",
            schema_version=SCHEMA_VERSION,
            model=state.get("model") or "",
            max_workers=state["max_workers"],
            operators=list_operator_names(),
            stats=ex.stats if ex else {},
        )

    @app.get("/v1/operators")
    def operators() -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "operators": list_operator_names(),
            "schemas": get_all_schemas(),
        }

    @app.get("/v1/stats")
    def stats() -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "stats": _exec().stats,
            "max_workers": state["max_workers"],
        }

    @app.post("/v1/ask")
    def ask(body: AskRequest) -> dict:
        result = _exec().run(body.id, body.query)
        return result.to_dict()

    @app.post("/v1/ask/batch")
    def ask_batch(body: BatchAskRequest) -> dict:
        if len(body.items) > _MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"batch size {len(body.items)} exceeds max {_MAX_BATCH_SIZE}",
            )
        pool: ThreadPoolExecutor = state["pool"]
        ex = _exec()

        # 校验 id 唯一
        ids = [it.id for it in body.items]
        if len(ids) != len(set(ids)):
            raise HTTPException(status_code=400, detail="batch items 中 id 必须唯一")

        futures = {
            pool.submit(ex.run, it.id, it.query): it.id for it in body.items
        }
        by_id: dict[str, dict] = {}
        for fut in as_completed(futures):
            rid = futures[fut]
            try:
                by_id[rid] = fut.result().to_dict()
            except Exception as exc:
                logger.exception("batch item failed id=%s", rid)
                by_id[rid] = empty_failed_response(
                    rid, f"{type(exc).__name__}: {exc}"
                ).to_dict()

        # 按请求顺序返回，保证前端可对位渲染
        results = [by_id[it.id] for it in body.items]
        return {
            "schema_version": SCHEMA_VERSION,
            "count": len(results),
            "results": results,
        }

    return app


def main() -> None:
    """uvicorn 启动入口：python -m spatio_temporal_agent.api.server"""
    import uvicorn

    host = os.environ.get("AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", "8000"))
    workers_env = os.environ.get("AGENT_MAX_WORKERS", str(_DEFAULT_MAX_WORKERS))
    logger.info("starting API host=%s port=%s max_workers=%s", host, port, workers_env)
    uvicorn.run(
        "spatio_temporal_agent.api.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
