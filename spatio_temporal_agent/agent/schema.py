"""B↔C 固定契约：请求 / 过程 / 响应结构。

约定一旦锁定，前端与评测脚本按此解析；变更需三方确认并升 version。
当前 schema_version = "1.0"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SCHEMA_VERSION = "1.0"


@dataclass
class ProcessStep:
    """单步执行过程（工具调用或内部阶段）。"""

    step: int
    type: str  # "kb_retrieve" | "tool_call" | "llm_answer" | "error"
    status: str  # "ok" | "error" | "skipped"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    message: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessTrace:
    """完整过程轨迹，供前端逐步渲染 / 评测统计。"""

    kb_context: str = ""
    kb_hits: list[dict] = field(default_factory=list)
    steps: list[ProcessStep] = field(default_factory=list)
    final_status: str = "pending"  # pending | completed | partial | failed

    def to_dict(self) -> dict:
        return {
            "kb_context": self.kb_context,
            "kb_hits": self.kb_hits,
            "steps": [s.to_dict() for s in self.steps],
            "final_status": self.final_status,
        }


@dataclass
class AgentResponse:
    """单条任务的标准响应（带 id）。"""

    id: str
    success: bool
    answer: str
    process: ProcessTrace
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "success": self.success,
            "answer": self.answer,
            "process": self.process.to_dict(),
            "error": self.error,
            "meta": self.meta,
        }


def empty_failed_response(
    request_id: str,
    error: str,
    *,
    elapsed_ms: float = 0.0,
    model: str = "",
) -> AgentResponse:
    """构造统一失败响应，保证结构始终完整。"""
    process = ProcessTrace(final_status="failed")
    process.steps.append(
        ProcessStep(
            step=1,
            type="error",
            status="error",
            name="executor",
            message=error,
        )
    )
    return AgentResponse(
        id=request_id,
        success=False,
        answer="",
        process=process,
        error=error,
        meta={"elapsed_ms": round(elapsed_ms, 2), "model": model},
    )
