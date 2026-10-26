"""离线演示脚本：不依赖 LLM，直接演示算子调用流程和知识库检索。

用于：
- 中期检查前快速验证算子和知识库逻辑正确性
- 展示"业务流程"结构（无需真实 LLM 也能跑通）

用法：
    python -m spatio_temporal_agent.demo_offline
"""

import json

from .knowledge_base import KnowledgeBase
from .operators.registry import call_operator, list_operator_names


def demo_operators() -> None:
    print("【演示1】算子直接调用")

    poly = [
        [31.20, 121.40], [31.20, 121.50],
        [31.30, 121.50], [31.30, 121.40],
    ]

    cases = [
        # ── 空间基础算子 ──
        ("haversine_distance", dict(lat1=31.2304, lon1=121.4737, lat2=39.9042, lon2=116.4074)),
        ("bearing", dict(lat1=31.2304, lon1=121.4737, lat2=39.9042, lon2=116.4074)),
        ("destination_point", dict(lat=31.2304, lon=121.4737, distance_m=1000, bearing_deg=90)),
        ("point_in_polygon", dict(lat=31.25, lon=121.45, polygon=poly)),
        ("polygon_area", dict(polygon=poly)),
        ("bounding_box_overlap", dict(
            bbox1={"min_lat": 31.0, "max_lat": 32.0, "min_lon": 121.0, "max_lon": 122.0},
            bbox2={"min_lat": 31.5, "max_lat": 32.5, "min_lon": 121.5, "max_lon": 122.5},
        )),
        ("buffer_circle", dict(lat=31.2304, lon=121.4737, radius_m=5000)),

        # ── 空间新算子 ──
        ("nearest_point", dict(
            target_lat=31.23, target_lon=121.47,
            candidates=[
                {"lat": 31.2304, "lon": 121.4737, "label": "上海·外滩"},
                {"lat": 39.9042, "lon": 116.4074, "label": "北京·故宫"},
                {"lat": 23.1291, "lon": 113.2644, "label": "广州·天河"},
            ],
        )),
        ("polygon_centroid", dict(polygon=poly)),
        ("line_length", dict(line=[[31.20, 121.40], [31.25, 121.45], [31.30, 121.50]])),
        ("angle_between", dict(
            origin=[31.23, 121.47],
            a=[31.30, 121.47],
            b=[31.23, 121.55],
        )),

        # ── 时间算子 ──
        ("temporal_distance", dict(
            t1="2026-10-18T08:00:00+08:00",
            t2="2026-10-18T18:30:00+08:00",
        )),
        ("timezone_convert", dict(
            timestamp="2026-10-18T08:00:00",
            from_tz="UTC",
            to_tz="Asia/Shanghai",
        )),
        ("time_range_filter", dict(
            events=[
                {"time": "2026-10-18T07:00:00+08:00", "place": "上海·外滩", "action": "日出观测"},
                {"time": "2026-10-18T12:00:00+08:00", "place": "上海·中心", "action": "巡检"},
                {"time": "2026-10-18T20:00:00+08:00", "place": "上海·外滩", "action": "夜景拍摄"},
                {"time": "2026-10-19T09:00:00+08:00", "place": "北京", "action": "出差"},
            ],
            start="2026-10-18T00:00:00+08:00",
            end="2026-10-18T23:59:59+08:00",
        )),
        ("time_series_resample", dict(
            times=[
                "2026-10-18T08:00:00+08:00",
                "2026-10-18T09:00:00+08:00",
                "2026-10-18T10:00:00+08:00",
                "2026-10-18T11:00:00+08:00",
            ],
            values=[20, 22, 24, 26],
            interval_hours=4,
        )),
    ]
    for name, args in cases:
        result = call_operator(name, **args)
        print(f"  {name}({json.dumps(args, ensure_ascii=False)})")
        print(f"    → {result.get('explanation') or result}")
        print()


def demo_knowledge_base() -> None:
    print("【演示2】领域知识库检索")
    kb = KnowledgeBase()
    queries = [
        "计算两个城市之间的距离",
        "判断某个点是否在多边形内部",
        "生成缓冲区",
    ]
    for q in queries:
        print(f"  查询：{q}")
        ctx = kb.build_context(q, top_k=2)
        print(f"  检索结果：\n{ctx}")
        print()


def demo_error_handling() -> None:
    print("【演示3】错漏案例（业务容错）")
    bad = call_operator("haversine_distance", lat1="abc", lon1=1, lat2=2, lon2=3)
    print(f"  参数缺失/错误：{bad}")
    unknown = call_operator("nonexistent_operator", foo=1)
    print(f"  未知算子：{unknown}")
    print()
    print(f"  已注册算子：{list_operator_names()}")


def main() -> None:
    print("=" * 60)
    print("  时空地理计算智能体 · 离线演示")
    print("=" * 60)
    demo_operators()
    demo_knowledge_base()
    demo_error_handling()


if __name__ == "__main__":
    main()
