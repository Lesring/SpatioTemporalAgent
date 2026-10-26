"""离线契约冒烟：不调用真实 LLM，验证 B↔C 响应结构。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from .agent.schema import SCHEMA_VERSION, AgentResponse, ProcessStep, ProcessTrace
from .api.server import create_app


class _FakeExecutor:
    llm = type("L", (), {"model": "fake-model"})()

    def __init__(self):
        self.stats = {"total": 0, "success": 0, "failed": 0}

    def run(self, request_id: str, query: str) -> AgentResponse:
        self.stats["total"] += 1
        self.stats["success"] += 1
        process = ProcessTrace(
            kb_context="demo",
            kb_hits=[{"title": "t", "description": "d", "keywords": ["k"]}],
            final_status="completed",
            steps=[
                ProcessStep(1, "kb_retrieve", "ok", "KnowledgeBase.search", message="ok"),
                ProcessStep(
                    2,
                    "tool_call",
                    "ok",
                    "haversine_distance",
                    args={"lat1": 1, "lon1": 2, "lat2": 3, "lon2": 4},
                    result={"success": True, "result": 100},
                    message="ok",
                    elapsed_ms=1.0,
                ),
                ProcessStep(3, "llm_answer", "ok", "final_answer", message="ok"),
            ],
        )
        return AgentResponse(
            id=request_id,
            success=True,
            answer=f"echo:{query}",
            process=process,
            error=None,
            meta={"elapsed_ms": 1.0, "model": "fake-model", "tool_call_count": 1,
                  "operator_names": ["haversine_distance"]},
        )


def main() -> None:
    app = create_app(executor=_FakeExecutor(), max_workers=2)
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200, h.text
        assert h.json()["schema_version"] == SCHEMA_VERSION

        one = client.post("/v1/ask", json={"id": "1", "query": "hi"})
        assert one.status_code == 200, one.text
        body = one.json()
        assert body["id"] == "1"
        assert body["schema_version"] == SCHEMA_VERSION
        assert "process" in body and "steps" in body["process"]
        assert body["answer"].startswith("echo:")

        batch = client.post(
            "/v1/ask/batch",
            json={"items": [{"id": "a", "query": "q1"}, {"id": "b", "query": "q2"}]},
        )
        assert batch.status_code == 200, batch.text
        results = batch.json()["results"]
        assert [r["id"] for r in results] == ["a", "b"]

    print("contract smoke OK")


if __name__ == "__main__":
    main()
