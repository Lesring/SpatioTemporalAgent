"""基本地理信息算子实现。

每个算子返回 dict，包含：
- success: bool
- result: 具体计算结果（数值或描述）
- explanation: 人类可读的结果说明

分为两类：
- 空间算子：距离 / 方位 / 多边形 / 缓冲区 等
- 时间算子：时间间隔 / 时区换算 / 时段过滤 / 时序重采样
"""

import math
import time as _time
from datetime import datetime, timedelta
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _to_float(value: Any) -> float:
    """将参数值安全转换为 float。"""
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"参数 {value!r} 无法转换为数值: {e}") from e


def haversine_distance(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> dict:
    """计算两点之间的 Haversine 距离（单位：米）。

    参数：
        lat1, lon1: 第一个点的纬度、经度（十进制度数）
        lat2, lon2: 第二个点的纬度、经度（十进制度数）
    """
    lat1f, lon1f = _to_float(lat1), _to_float(lon1)
    lat2f, lon2f = _to_float(lat2), _to_float(lon2)

    R = 6_371_000.0  # 地球平均半径（米）
    phi1, phi2 = math.radians(lat1f), math.radians(lat2f)
    d_phi = math.radians(lat2f - lat1f)
    d_lambda = math.radians(lon2f - lon1f)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_m = R * c

    return {
        "success": True,
        "operator": "haversine_distance",
        "result": round(distance_m, 2),
        "unit": "米",
        "explanation": f"两点间距离为 {distance_m / 1000:.2f} 公里（{distance_m:.2f} 米）",
    }


def bearing(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> dict:
    """计算从点1到点2的方位角（北起顺时针，0-360度）。"""
    lat1f, lon1f = _to_float(lat1), _to_float(lon1)
    lat2f, lon2f = _to_float(lat2), _to_float(lon2)

    phi1, phi2 = math.radians(lat1f), math.radians(lat2f)
    d_lambda = math.radians(lon2f - lon1f)

    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    angle = math.atan2(y, x)
    bearing_deg = (math.degrees(angle) + 360) % 360

    return {
        "success": True,
        "operator": "bearing",
        "result": round(bearing_deg, 2),
        "unit": "度（北起顺时针）",
        "explanation": f"从点1到点2的方位角为 {bearing_deg:.2f} 度",
    }


def destination_point(lat: Any, lon: Any, distance_m: Any, bearing_deg: Any) -> dict:
    """根据起点、距离和方位角，计算终点坐标。"""
    latf = _to_float(lat)
    lonf = _to_float(lon)
    dfm = _to_float(distance_m)
    b = math.radians(_to_float(bearing_deg))

    R = 6_371_000.0
    dr = dfm / R
    phi1 = math.radians(latf)
    lambda1 = math.radians(lonf)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(dr) + math.cos(phi1) * math.sin(dr) * math.cos(b)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(b) * math.sin(dr) * math.cos(phi1),
        math.cos(dr) - math.sin(phi1) * math.sin(phi2),
    )
    result_lat = math.degrees(phi2)
    result_lon = ((math.degrees(lambda2) + 540) % 360) - 180

    return {
        "success": True,
        "operator": "destination_point",
        "result": {"lat": round(result_lat, 6), "lon": round(result_lon, 6)},
        "explanation": f"从 ({latf}, {lonf}) 出发，沿 {bearing_deg}° 方向走 {dfm} 米，到达 ({result_lat:.4f}, {result_lon:.4f})",
    }


def point_in_polygon(lat: Any, lon: Any, polygon: list) -> dict:
    """判断点是否在多边形内（射线法）。

    polygon 格式：[[lat, lon], [lat, lon], ...]
    """
    latf = _to_float(lat)
    lonf = _to_float(lon)

    try:
        poly = [[_to_float(p[0]), _to_float(p[1])] for p in polygon]
    except (IndexError, TypeError) as e:
        return {"success": False, "error": f"多边形坐标格式错误: {e}"}

    # 射线法（以经度为 x，纬度为 y）
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i][0], poly[i][1]
        yj, xj = poly[j][0], poly[j][1]
        if ((xi > lonf) != (xj > lonf)) and (
            latf < (yj - yi) * (lonf - xi) / (xj - xi + 1e-12) + yi
        ):
            inside = not inside
        j = i

    return {
        "success": True,
        "operator": "point_in_polygon",
        "result": inside,
        "explanation": f"点 ({latf}, {lonf}) {'在' if inside else '不在'}多边形内部",
    }


def polygon_area(polygon: list) -> dict:
    """计算多边形面积（基于经纬度近似，单位：平方米）。

    polygon 格式：[[lat, lon], [lat, lon], ...]
    """
    try:
        poly = [[_to_float(p[0]), _to_float(p[1])] for p in polygon]
    except (IndexError, TypeError) as e:
        return {"success": False, "error": f"多边形坐标格式错误: {e}"}

    if len(poly) < 3:
        return {"success": False, "error": "多边形至少需要3个顶点"}

    # 使用球面近似：将经纬度转换为米（局部平面近似）
    avg_lat = sum(p[0] for p in poly) / len(poly)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(avg_lat))

    # 鞋带公式
    n = len(poly)
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        xi = poly[i][1] * m_per_deg_lon
        yi = poly[i][0] * m_per_deg_lat
        xj = poly[j][1] * m_per_deg_lon
        yj = poly[j][0] * m_per_deg_lat
        area2 += xi * yj - xj * yi
    area_m2 = abs(area2) / 2.0

    return {
        "success": True,
        "operator": "polygon_area",
        "result": round(area_m2, 2),
        "unit": "平方米",
        "explanation": f"多边形面积约为 {area_m2 / 1e6:.4f} 平方公里（{area_m2:.2f} 平方米）",
    }


def bounding_box_overlap(
    bbox1: dict, bbox2: dict
) -> dict:
    """判断两个边界框是否重叠。

    每个 bbox 格式：{"min_lat": .., "max_lat": .., "min_lon": .., "max_lon": ..}
    返回重叠区域或 None。
    """
    try:
        min_lat = max(_to_float(bbox1["min_lat"]), _to_float(bbox2["min_lat"]))
        max_lat = min(_to_float(bbox1["max_lat"]), _to_float(bbox2["max_lat"]))
        min_lon = max(_to_float(bbox1["min_lon"]), _to_float(bbox2["min_lon"]))
        max_lon = min(_to_float(bbox1["max_lon"]), _to_float(bbox2["max_lon"]))
    except (KeyError, TypeError) as e:
        return {"success": False, "error": f"边界框字段缺失或格式错误: {e}"}

    overlaps = min_lat < max_lat and min_lon < max_lon

    if not overlaps:
        return {
            "success": True,
            "operator": "bounding_box_overlap",
            "result": False,
            "explanation": "两个区域无重叠",
        }

    overlap_bbox = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
    }
    return {
        "success": True,
        "operator": "bounding_box_overlap",
        "result": True,
        "overlap_bbox": overlap_bbox,
        "explanation": f"两个区域重叠，重叠范围: {overlap_bbox}",
    }


def buffer_circle(lat: Any, lon: Any, radius_m: Any) -> dict:
    """生成以某点为圆心、指定半径的圆形缓冲区的边界框。"""
    latf = _to_float(lat)
    lonf = _to_float(lon)
    rf = _to_float(radius_m)

    delta_lat = rf / 111_320.0
    delta_lon = rf / (111_320.0 * max(math.cos(math.radians(latf)), 1e-10))

    bbox = {
        "min_lat": latf - delta_lat,
        "max_lat": latf + delta_lat,
        "min_lon": lonf - delta_lon,
        "max_lon": lonf + delta_lon,
    }
    return {
        "success": True,
        "operator": "buffer_circle",
        "result": bbox,
        "explanation": f"以 ({latf}, {lonf}) 为圆心，半径 {rf} 米的缓冲区边界框为 {bbox}",
    }


# ── 新增空间算子 ─────────────────────────────────────────────

def nearest_point(target_lat: Any, target_lon: Any, candidates: list) -> dict:
    """在一组候选点中找距离目标最近的一个。

    candidates 格式：[{"lat": .., "lon": .., "label": 可选}, ...]
    返回最近点的标签/坐标与距离。
    """
    tlat, tlon = _to_float(target_lat), _to_float(target_lon)
    best = None
    best_dist = None
    for c in candidates:
        try:
            lat = _to_float(c["lat"]) if isinstance(c, dict) else _to_float(c[0])
            lon = _to_float(c["lon"]) if isinstance(c, dict) else _to_float(c[1])
        except (KeyError, IndexError, TypeError) as e:
            continue
        d = _haversine(tlat, tlon, lat, lon)
        if best_dist is None or d < best_dist:
            best_dist = d
            label = c.get("label") if isinstance(c, dict) else None
            best = {"lat": lat, "lon": lon, "label": label, "distance_m": round(d, 2)}
    if best is None:
        return {"success": False, "error": "候选点列表为空或格式错误"}
    label = best.get("label") or f"({best['lat']}, {best['lon']})"
    return {
        "success": True,
        "operator": "nearest_point",
        "result": best,
        "explanation": f"最近点是 {label}，距离目标 {best['distance_m']} 米",
    }


def polygon_centroid(polygon: list) -> dict:
    """计算多边形顶点集的质心（算术平均，非面积加权）。"""
    try:
        poly = [[_to_float(p[0]), _to_float(p[1])] for p in polygon]
    except (IndexError, TypeError) as e:
        return {"success": False, "error": f"多边形坐标格式错误: {e}"}
    if not poly:
        return {"success": False, "error": "多边形顶点列表为空"}
    lat = sum(p[0] for p in poly) / len(poly)
    lon = sum(p[1] for p in poly) / len(poly)
    return {
        "success": True,
        "operator": "polygon_centroid",
        "result": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "explanation": f"多边形质心（顶点算术平均）为 ({lat:.4f}, {lon:.4f})",
    }


def line_length(line: list) -> dict:
    """计算折线（按顺序的坐标串）总长度（米），用相邻点 Haversine 累加。

    line 格式：[[lat, lon], [lat, lon], ...]
    """
    try:
        pts = [[_to_float(p[0]), _to_float(p[1])] for p in line]
    except (IndexError, TypeError) as e:
        return {"success": False, "error": f"折线坐标格式错误: {e}"}
    if len(pts) < 2:
        return {"success": False, "error": "折线至少需要2个点"}
    total = sum(_haversine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]) for i in range(len(pts)-1))
    return {
        "success": True,
        "operator": "line_length",
        "result": round(total, 2),
        "unit": "米",
        "explanation": f"折线总长度约 {total/1000:.2f} 公里（{total:.2f} 米）",
    }


def angle_between(origin: list, a: list, b: list) -> dict:
    """计算以 origin 为顶点，射线 origin→a 与 origin→b 之间的夹角（度）。

    每个参数格式 [lat, lon]。
    """
    def _latlon(p):
        return _to_float(p[0]), _to_float(p[1])
    olat, olon = _latlon(origin)
    alat, alon = _latlon(a)
    blat, blon = _latlon(b)
    ba = _bearing(olat, olon, alat, alon)
    bb = _bearing(olat, olon, blat, blon)
    diff = abs(ba - bb) % 360
    angle = min(diff, 360 - diff)
    return {
        "success": True,
        "operator": "angle_between",
        "result": round(angle, 2),
        "unit": "度",
        "explanation": f"以 origin 为顶点，a 与 b 两射线的夹角为 {angle:.2f} 度",
    }


# ── 时间算子 ─────────────────────────────────────────────────

def temporal_distance(t1: str, t2: str) -> dict:
    """计算两个 ISO 8601 时间戳之间的时间间隔。

    支持带时区偏移的字符串（如 '2026-10-18T08:00:00+08:00'）或 naive 时间。
    返回小时数与人类可读描述。
    """
    try:
        d1 = _parse_iso(t1)
        d2 = _parse_iso(t2)
    except ValueError as e:
        return {"success": False, "error": f"时间解析失败: {e}"}
    naive = not (d1.tzinfo or d2.tzinfo)
    delta = d2 - d1
    if naive:
        delta = delta  # 按本地时间差处理
    total_seconds = delta.total_seconds()
    hours = total_seconds / 3600
    sign = "" if total_seconds >= 0 else "-"
    abs_s = abs(total_seconds)
    d, rem = int(abs_s // 86400), abs_s % 86400
    h, rem = int(rem // 3600), rem % 3600
    m, s = int(rem // 60), int(rem % 60)
    return {
        "success": True,
        "operator": "temporal_distance",
        "result": round(hours, 4),
        "unit": "小时",
        "explanation": f"两个时间点相差 {sign}{d}天 {h}时 {m}分 {s}秒（约 {hours:.2f} 小时）",
    }


def timezone_convert(timestamp: str, from_tz: str, to_tz: str) -> dict:
    """将某时间戳从 from_tz 时区换算到 to_tz 时区。

    timestamp: ISO 8601（可带时区；若不带，则视为 from_tz 本地时间）
    from_tz / to_tz: IANA 时区名，如 'Asia/Shanghai', 'Europe/London'
    """
    try:
        dt = _parse_iso(timestamp)
        tz_from = ZoneInfo(from_tz)
        tz_to = ZoneInfo(to_tz)
    except (ValueError, ZoneInfoNotFoundError) as e:
        return {"success": False, "error": f"时间或时区无效: {e}"}
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz_from)
    converted = dt.astimezone(tz_to)
    return {
        "success": True,
        "operator": "timezone_convert",
        "result": converted.isoformat(),
        "explanation": f"{timestamp} 在 {from_tz} 相当于 {to_tz} 的 {converted.strftime('%Y-%m-%d %H:%M:%S %Z')}",
    }


def time_range_filter(events: list, start: str, end: str) -> dict:
    """从带时间戳的事件列表中，筛选出落在 [start, end] 时间范围内的条目。

    events 格式：[{"time": "ISO时间戳", ...其他字段}, ...]
    start / end: ISO 8601 时间戳
    """
    try:
        s = _parse_iso(start)
        e = _parse_iso(end)
    except ValueError as ex:
        return {"success": False, "error": f"时间范围解析失败: {ex}"}
    in_range = []
    for ev in events:
        try:
            t = _parse_iso(ev.get("time"))
        except (ValueError, AttributeError):
            continue
        if t is None:
            continue
        if s.tzinfo is not None and t.tzinfo is None:
            t = t.replace(tzinfo=s.tzinfo)
        if s <= t <= e:
            in_range.append(ev)
    return {
        "success": True,
        "operator": "time_range_filter",
        "result": in_range,
        "count": len(in_range),
        "explanation": f"共 {len(events)} 条事件，其中 {len(in_range)} 条落在 [{start}, {end}] 范围内",
    }


def time_series_resample(times: list, values: list, interval_hours: Any) -> dict:
    """对时序数据按固定间隔（小时）重采样，输出每个桶内的均值。

    times: ISO 时间戳列表（升序）；values: 对应数值列表；interval_hours: 桶宽（小时）
    返回每个时间桶的 [桶起, 桶止, 均值]。
    """
    if len(times) != len(values):
        return {"success": False, "error": "times 与 values 长度不一致"}
    if not times:
        return {"success": False, "error": "时序数据为空"}
    try:
        parsed = [( _parse_iso(t), float(v)) for t, v in zip(times, values)]
        iv = _to_float(interval_hours)
    except (ValueError, TypeError) as e:
        return {"success": False, "error": f"参数解析失败: {e}"}
    parsed.sort(key=lambda x: x[0])
    buckets: dict[int, list] = {}
    for t, v in parsed:
        # 用 Unix 时间戳对齐桶（统一用 int 避免浮点字符串问题）
        bucket_size_s = int(round(iv * 3600))
        key = int(t.timestamp() // bucket_size_s) * bucket_size_s
        buckets.setdefault(key, []).append(v)
    result = []
    for k in sorted(buckets):
        vals = buckets[k]
        result.append({
            "bucket_start_utc_unix": k,
            "avg": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        })
    return {
        "success": True,
        "operator": "time_series_resample",
        "result": result,
        "explanation": f"按 {iv} 小时重采样，得到 {len(result)} 个时间桶的均值",
    }


# ── 内部辅助函数 ─────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    """返回两点距离（米），复用 haversine_distance 的核心。"""
    return haversine_distance(lat1, lon1, lat2, lon2)["result"]


def _bearing(lat1, lon1, lat2, lon2):
    """返回方位角（度），复用 bearing 核心。"""
    return bearing(lat1, lon1, lat2, lon2)["result"]


def _parse_iso(s: str) -> datetime:
    """解析 ISO 8601 时间戳，兼容带/不带时区的格式。"""
    s = s.strip()
    # Python 3.11+ fromisoformat 支持 Z；旧版本替换
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)
