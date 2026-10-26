"""业务流程准确率评测脚本。

用一组固定测试用例，统计智能体在"自然语言 → 算子选择 + 参数提取 → 结果"
这条业务流程上的准确率，输出可量化的指标（对齐题目评分要点"业务流程准确率"）。

指标定义：
- 算子选择命中率：期望算子是否出现在实际调用链中
- 完整流程成功率：最终回答 success 且调用链正确
- 地名自动解析率：涉及地名的用例是否成功调用了 resolve_place

用法：
    python -m spatio_temporal_agent.evaluate
"""

from dataclasses import dataclass, field
from typing import Optional

from .agent import LLMClient, SpatioTemporalAgent


@dataclass
class Case:
    """单条评测用例。"""
    query: str
    expect_operators: list[str]      # 期望被调用的算子（按包含关系判断，不要求顺序）
    expect_success: bool = True      # 期望最终流程成功
    expect_place_resolve: bool = False  # 期望触发了地名解析
    note: str = ""


# 固定评测集（空间 + 时间 + 多步骤 + 地名解析）
EVAL_CASES: list[Case] = [
    Case(
        query="计算上海到杭州的距离",
        expect_operators=["resolve_place", "haversine_distance"],
        expect_place_resolve=True,
        note="地名→坐标→距离 多步骤",
    ),
    Case(
        query="从北京到成都的方位角是多少",
        expect_operators=["resolve_place", "bearing"],
        expect_place_resolve=True,
        note="地名→方位角 多步骤",
    ),
    Case(
        query="以上海为圆心，生成半径5000米的缓冲区",
        expect_operators=["resolve_place", "buffer_circle"],
        expect_place_resolve=True,
    ),
    Case(
        query="判断点(31.25, 121.45)是否在多边形[[31.2,121.4],[31.2,121.5],[31.3,121.5],[31.3,121.4]]内部",
        expect_operators=["point_in_polygon"],
        note="纯坐标，无需地名解析",
    ),
    Case(
        query="计算多边形[[31.2,121.4],[31.2,121.5],[31.3,121.5],[31.3,121.4]]的面积",
        expect_operators=["polygon_area"],
    ),
    Case(
        query="2026-10-18T08:00:00+08:00 到 2026-10-18T20:00:00+08:00 相隔多少小时",
        expect_operators=["temporal_distance"],
        note="时间算子",
    ),
    Case(
        query="把北京时间 2026-10-18 08:00 换算成伦敦时间",
        expect_operators=["timezone_convert"],
        note="时区换算",
    ),
    Case(
        query="判断区域[31.0,32.0,121.0,122.0]和区域[31.5,32.5,121.5,122.5]是否重叠",
        expect_operators=["bounding_box_overlap"],
        note="边界框重叠",
    ),
]


def _run_case(agent: SpatioTemporalAgent, case: Case) -> dict:
    """跑单条用例；LLM/网络失败时标记失败但不抛出，保证整体统计不被中断。"""
    try:
        result = agent.ask(case.query)
    except Exception as e:
        return {
            "query": case.query,
            "called": [],
            "all_ops_ok": False,
            "operator_hit": False,
            "place_resolve_hit": False,
            "flow_success": False,
            "answer": f"[评测中断] {type(e).__name__}: {e}",
            "note": case.note,
        }

    called_ops = [tc["operator"] for tc in result["tool_calls_made"]]
    answer = result.get("answer") or ""

    # 成功判定：无 success=False 的算子调用 + 有回答
    all_ok = all(
        tc["result"].get("success") for tc in result["tool_calls_made"]
    ) if result["tool_calls_made"] else True

    op_hit = all(op in called_ops for op in case.expect_operators)
    place_hit = ("resolve_place" in called_ops) if case.expect_place_resolve else True

    return {
        "query": case.query,
        "called": called_ops,
        "all_ops_ok": all_ok,
        "operator_hit": op_hit,
        "place_resolve_hit": place_hit,
        "flow_success": bool(answer.strip()) and all_ok and op_hit,
        "answer": answer,
        "note": case.note,
    }


def run_evaluation(
    agent: SpatioTemporalAgent,
    cases: list[Case] = EVAL_CASES,
    verbose: bool = True,
) -> dict:
    """跑完整评测集，输出准确率统计。"""
    results = [_run_case(agent, c) for c in cases]

    total = len(results)
    op_hits = sum(r["operator_hit"] for r in results)
    flow_success = sum(r["flow_success"] for r in results)
    # 地名解析准确率只在"期望触发地名解析"的用例上统计
    place_pairs = [(r, c) for r, c in zip(results, cases) if c.expect_place_resolve]
    place_hits = sum(r["place_resolve_hit"] for r, c in place_pairs if c.expect_place_resolve)
    place_total = len(place_pairs)

    metrics = {
        "total_cases": total,
        "operator_selection_accuracy": round(op_hits / total, 3) if total else 0,
        "place_resolution_accuracy": (
            round(place_hits / place_total, 3) if place_total else 1.0
        ),
        "overall_flow_success_rate": round(flow_success / total, 3) if total else 0,
    }

    if verbose:
        print("=" * 70)
        print("  业务流程准确率评测报告")
        print("=" * 70)
        for i, r in enumerate(results, 1):
            flag = "✅" if r["flow_success"] else "❌"
            print(f"\n{i}. {flag} {r['query'][:40]}...")
            print(f"   实际调用: {r['called']}")
            print(f"   算子命中: {'是' if r['operator_hit'] else '否'} | "
                  f"地名解析: {'是' if r['place_resolve_hit'] else 'n/a'} | "
                  f"流程成功: {'是' if r['flow_success'] else '否'}")
        print("\n" + "-" * 70)
        print(f"  用例总数:            {metrics['total_cases']}")
        print(f"  算子选择准确率:      {metrics['operator_selection_accuracy']*100:.1f}%")
        print(f"  地名自动解析准确率:  {metrics['place_resolution_accuracy']*100:.1f}%")
        print(f"  整体流程成功率:      {metrics['overall_flow_success_rate']*100:.1f}%")
        print("-" * 70)

    return {"metrics": metrics, "results": results}


def main() -> None:
    llm = LLMClient.from_config_file("spatio_temporal_agent/config.json")
    print(f"使用模型: {llm.model} @ {llm.api_base}")
    agent = SpatioTemporalAgent(llm=llm)
    run_evaluation(agent)


if __name__ == "__main__":
    main()
