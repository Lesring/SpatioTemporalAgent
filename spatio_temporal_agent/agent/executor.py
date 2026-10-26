"""任务执行器：B 层对外唯一入口。

职责：
1. 接收带 id 的自然语言任务
2. 调用 SpatioTemporalAgent 完成编排
3. 输出 schema 固定的 AgentResponse
4. 记录状态、耗时、异常
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Optional

from ..knowledge_base import KnowledgeBase
from .llm_client import LLMClient
from .logging_utils import get_logger
from .schema import (
    AgentResponse,
    ProcessStep,
    ProcessTrace,
    empty_failed_response,
)
from .spatio_temporal_agent import SpatioTemporalAgent

logger = get_logger()


class TaskExecutor:
    """带状态管理的任务执行器（可被 HTTP / CLI / 评测复用）。"""

    def __init__(
        self,
        llm: LLMClient,
        kb: Optional[KnowledgeBase] = None,
        max_tool_rounds: int = 8,
        max_tool_calls: int = 20,
    ):
        self.llm = llm
        self.agent = SpatioTemporalAgent(
            llm=llm,
            kb=kb,
            max_tool_rounds=max_tool_rounds,
            max_tool_calls=max_tool_calls,
        )
        self._stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def run(self, request_id: str, query: str) -> AgentResponse:
        """执行单条任务，永远返回结构完整的 AgentResponse。"""
        started = time.perf_counter()
        self._stats["total"] += 1
        q = (query or "").strip()
        rid = str(request_id or "").strip() or "anonymous"

        if not q:
            self._stats["failed"] += 1
            return empty_failed_response(
                rid,
                "query 不能为空",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                model=self.llm.model,
            )

        logger.info("task start id=%s query=%s", rid, q[:120])
        try:
            raw = self.agent.ask(q)
            response = self._build_response(rid, raw, started)
            if response.success:
                self._stats["success"] += 1
            else:
                self._stats["failed"] += 1
            logger.info(
                "task end id=%s success=%s status=%s elapsed_ms=%.1f",
                rid,
                response.success,
                response.process.final_status,
                response.meta.get("elapsed_ms", 0),
            )
            return response
        except Exception as exc:
            self._stats["failed"] += 1
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception("task failed id=%s: %s", rid, exc)
            resp = empty_failed_response(
                rid,
                f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
                model=getattr(self.llm, "model", ""),
            )
            resp.meta["traceback"] = traceback.format_exc()[-2000:]
            return resp

    def _build_response(
        self,
        request_id: str,
        raw: dict[str, Any],
        started: float,
    ) -> AgentResponse:
        elapsed = (time.perf_counter() - started) * 1000
        tool_calls = raw.get("tool_calls_made") or []
        kb_context = raw.get("kb_context") or ""
        kb_hits = raw.get("kb_hits") or []
        answer = raw.get("answer") or ""

        process = ProcessTrace(
            kb_context=kb_context,
            kb_hits=kb_hits,
        )

        step_i = 1
        process.steps.append(
            ProcessStep(
                step=step_i,
                type="kb_retrieve",
                status="ok" if (kb_context and "暂无相关" not in kb_context) else "ok",
                name="KnowledgeBase.search",
                result={"preview": kb_context[:500]},
                message="领域知识检索完成",
            )
        )
        step_i += 1

        any_tool_error = False
        for tc in tool_calls:
            result = tc.get("result") or {}
            ok = bool(result.get("success", True))
            if not ok:
                any_tool_error = True
            process.steps.append(
                ProcessStep(
                    step=step_i,
                    type="tool_call",
                    status="ok" if ok else "error",
                    name=tc.get("operator", ""),
                    args=tc.get("args") or {},
                    result=result,
                    message=result.get("explanation")
                    or result.get("error")
                    or "",
                    elapsed_ms=float(tc.get("elapsed_ms") or 0),
                )
            )
            step_i += 1

        process.steps.append(
            ProcessStep(
                step=step_i,
                type="llm_answer",
                status="ok" if answer else "error",
                name="final_answer",
                message="生成最终自然语言回答" if answer else "未生成回答",
            )
        )

        if not answer:
            process.final_status = "failed"
            success = False
            error = "未能生成最终回答"
        elif any_tool_error:
            process.final_status = "partial"
            success = True
            error = None
        elif tool_calls or answer:
            process.final_status = "completed"
            success = True
            error = None
        else:
            process.final_status = "failed"
            success = False
            error = "空结果"

        return AgentResponse(
            id=request_id,
            success=success,
            answer=answer,
            process=process,
            error=error,
            meta={
                "elapsed_ms": round(elapsed, 2),
                "model": self.llm.model,
                "tool_call_count": len(tool_calls),
                "operator_names": [tc.get("operator") for tc in tool_calls],
            },
        )
