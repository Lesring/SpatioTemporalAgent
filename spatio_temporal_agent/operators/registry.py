"""算子 schema 注册表。

将 Python 算子函数映射为 LLM function calling 所需的 JSON schema 描述。
每个 schema 包含：
- name: 算子名（对应 Python 函数名）
- description: 功能描述（帮助 LLM 判断何时调用）
- parameters: JSON Schema 参数描述
"""

from typing import Any, Callable, Optional

from . import geo_operators

# 算子注册表：name -> (python_function, openai_function_schema)
_OPERATOR_REGISTRY: dict[str, tuple[Callable, dict]] = {}


def _coerce_value(value: Any) -> Any:
    """把 LLM 传参可能出现的"字符串形式"自动解析为结构化类型。

    处理两类常见情况：
    - 字符串里其实是 JSON 数组/对象（如 "'[[31.2,121.4],...]"）→ 解析为 list/dict
    - 数值以字符串形式传入（如 "121.4"）→ 解析为 float

    若解析失败，原样返回，由算子内部再做校验。
    """
    if isinstance(value, str):
        s = value.strip()
        if s and s[0] in "[{":
            try:
                import json as _json
                return _json.loads(s)
            except Exception:
                return value
        try:
            return float(s)
        except ValueError:
            return value
    return value


def _coerce_args(kwargs: dict) -> dict:
    """对算子入参做一层类型兜底。"""
    return {k: _coerce_value(v) for k, v in kwargs.items()}


def register(
    name: str,
    func: Callable,
    description: str,
    parameters: dict,
) -> None:
    _OPERATOR_REGISTRY[name] = (func, {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    })


# ── 注册各算子 ──────────────────────────────────────────────

register(
    "haversine_distance",
    geo_operators.haversine_distance,
    "计算两个地理坐标点之间的球面距离（单位：米）。适用于求两点间距离。",
    {
        "type": "object",
        "properties": {
            "lat1": {"type": "number", "description": "第一个点的纬度（十进制度）"},
            "lon1": {"type": "number", "description": "第一个点的经度（十进制度）"},
            "lat2": {"type": "number", "description": "第二个点的纬度（十进制度）"},
            "lon2": {"type": "number", "description": "第二个点的经度（十进制度）"},
        },
        "required": ["lat1", "lon1", "lat2", "lon2"],
    },
)

register(
    "bearing",
    geo_operators.bearing,
    "计算从起点到终点的方位角（北起顺时针，0-360度）。",
    {
        "type": "object",
        "properties": {
            "lat1": {"type": "number", "description": "起点纬度"},
            "lon1": {"type": "number", "description": "起点经度"},
            "lat2": {"type": "number", "description": "终点纬度"},
            "lon2": {"type": "number", "description": "终点经度"},
        },
        "required": ["lat1", "lon1", "lat2", "lon2"],
    },
)

register(
    "destination_point",
    geo_operators.destination_point,
    "根据起点坐标、行进距离（米）和方位角（度）计算终点坐标。",
    {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "起点纬度"},
            "lon": {"type": "number", "description": "起点经度"},
            "distance_m": {"type": "number", "description": "行进距离（米）"},
            "bearing_deg": {"type": "number", "description": "方位角（北起顺时针，0-360度）"},
        },
        "required": ["lat", "lon", "distance_m", "bearing_deg"],
    },
)

register(
    "point_in_polygon",
    geo_operators.point_in_polygon,
    "判断一个点是否位于多边形内部（射线法）。多边形顶点列表按顺序给出。",
    {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "待判断点的纬度"},
            "lon": {"type": "number", "description": "待判断点的经度"},
            "polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "多边形顶点列表，每个顶点为 [纬度, 经度]",
            },
        },
        "required": ["lat", "lon", "polygon"],
    },
)

register(
    "polygon_area",
    geo_operators.polygon_area,
    "计算多边形（经纬度坐标）的近似面积（平方米），基于局部平面近似。",
    {
        "type": "object",
        "properties": {
            "polygon": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "description": "多边形顶点列表，每个顶点为 [纬度, 经度]",
            },
        },
        "required": ["polygon"],
    },
)

register(
    "bounding_box_overlap",
    geo_operators.bounding_box_overlap,
    "判断两个地理边界框是否重叠，并返回重叠区域（如有）。",
    {
        "type": "object",
        "properties": {
            "bbox1": {
                "type": "object",
                "properties": {
                    "min_lat": {"type": "number"},
                    "max_lat": {"type": "number"},
                    "min_lon": {"type": "number"},
                    "max_lon": {"type": "number"},
                },
                "description": "第一个边界框",
            },
            "bbox2": {
                "type": "object",
                "properties": {
                    "min_lat": {"type": "number"},
                    "max_lat": {"type": "number"},
                    "min_lon": {"type": "number"},
                    "max_lon": {"type": "number"},
                },
                "description": "第二个边界框",
            },
        },
        "required": ["bbox1", "bbox2"],
    },
)

register(
    "buffer_circle",
    geo_operators.buffer_circle,
    "以某点为圆心，生成指定半径（米）的圆形缓冲区的边界框。",
    {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "圆心纬度"},
            "lon": {"type": "number", "description": "圆心经度"},
            "radius_m": {"type": "number", "description": "缓冲区半径（米）"},
        },
        "required": ["lat", "lon", "radius_m"],
    },
)

register(
    "nearest_point",
    geo_operators.nearest_point,
    "在一组候选点中找距离目标点最近的一个，返回其坐标、标签与距离（米）。",
    {
        "type": "object",
        "properties": {
            "target_lat": {"type": "number", "description": "目标点纬度"},
            "target_lon": {"type": "number", "description": "目标点经度"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "label": {"type": "string"},
                    },
                    "required": ["lat", "lon"],
                },
                "description": "候选点列表",
            },
        },
        "required": ["target_lat", "target_lon", "candidates"],
    },
)

register(
    "polygon_centroid",
    geo_operators.polygon_centroid,
    "计算多边形顶点集的质心（顶点算术平均）。",
    {
        "type": "object",
        "properties": {
            "polygon": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "description": "多边形顶点列表，每个顶点为 [纬度, 经度]",
            },
        },
        "required": ["polygon"],
    },
)

register(
    "line_length",
    geo_operators.line_length,
    "计算折线（按顺序的坐标串）总长度（米），相邻点 Haversine 累加。",
    {
        "type": "object",
        "properties": {
            "line": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "description": "折线坐标串，每个点为 [纬度, 经度]，按行进顺序",
            },
        },
        "required": ["line"],
    },
)

register(
    "angle_between",
    geo_operators.angle_between,
    "计算以某点为顶点，到 a、b 两点的两条射线之间的夹角（度，0-180）。",
    {
        "type": "object",
        "properties": {
            "origin": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "顶点 [纬度, 经度]"},
            "a": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "射线a终点 [纬度, 经度]"},
            "b": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "射线b终点 [纬度, 经度]"},
        },
        "required": ["origin", "a", "b"],
    },
)

register(
    "temporal_distance",
    geo_operators.temporal_distance,
    "计算两个 ISO 8601 时间戳之间的时间间隔（小时），并给出人类可读描述。",
    {
        "type": "object",
        "properties": {
            "t1": {"type": "string", "description": "第一个时间戳（ISO 8601，可带时区）"},
            "t2": {"type": "string", "description": "第二个时间戳（ISO 8601，可带时区）"},
        },
        "required": ["t1", "t2"],
    },
)

register(
    "timezone_convert",
    geo_operators.timezone_convert,
    "将时间戳从一个 IANA 时区换算到另一个时区（如 Asia/Shanghai → Europe/London）。",
    {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string", "description": "ISO 8601 时间戳（不带时区时视为 from_tz 本地时间）"},
            "from_tz": {"type": "string", "description": "源时区（IANA 名，如 Asia/Shanghai）"},
            "to_tz": {"type": "string", "description": "目标时区（IANA 名，如 Europe/London）"},
        },
        "required": ["timestamp", "from_tz", "to_tz"],
    },
)

register(
    "time_range_filter",
    geo_operators.time_range_filter,
    "从带时间戳的事件列表中筛选出落在 [start, end] 时间范围内的条目。",
    {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {"type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]},
                "description": "事件列表，每个事件至少含 time 字段（ISO 时间戳）",
            },
            "start": {"type": "string", "description": "范围起始（ISO 8601）"},
            "end": {"type": "string", "description": "范围结束（ISO 8601）"},
        },
        "required": ["events", "start", "end"],
    },
)

register(
    "time_series_resample",
    geo_operators.time_series_resample,
    "对时序数据按固定间隔（小时）重采样，输出每个时间桶内数值的均值。",
    {
        "type": "object",
        "properties": {
            "times": {"type": "array", "items": {"type": "string"}, "description": "ISO 时间戳列表（升序）"},
            "values": {"type": "array", "items": {"type": "number"}, "description": "对应数值列表"},
            "interval_hours": {"type": "number", "description": "桶宽（小时）"},
        },
        "required": ["times", "values", "interval_hours"],
    },
)

# ── 地名解析算子（Gazetteer）────────────────────────────────
# 把 Geocoder 封装成可被 LLM 调用的算子。用户输入"上海"时，
# 智能体可调用 resolve_place 得到坐标，无需用户手打经纬度。
from ..geocoding import Geocoder  # 延迟导入避免循环依赖

_geocoder_instance: Optional["Geocoder"] = None


def _get_geocoder() -> "Geocoder":
    global _geocoder_instance
    if _geocoder_instance is None:
        _geocoder_instance = Geocoder()
    return _geocoder_instance


def _resolve_place(place: str, use_online: bool = True) -> dict:
    """把地名解析为坐标。本地词典命中则离线；未命中则 Nominatim 在线并写回。"""
    geo = _get_geocoder()
    result = geo.geocode(place, use_online=use_online, write_back=True)
    if not result.get("success", False):
        err_kind = result.get("error_kind")
        hint = {
            "not_found": "该地名确实无法解析（本地与在线均无）。可让用户提供该地的经纬度或更具体的地址。",
            "rate_limited": "Nominatim 返回 429 限速，属暂时性故障，直接告知用户稍后重试，**不要换写法反复尝试**。",
            "blocked": "Nominatim 持续拒绝本机 IP（403），属持续性限制，重试无效。可告知用户：在线地理编码暂不可用，请直接提供该地经纬度，或改用本地词典已收录的地名。",
            "network_error": "网络临时异常，直接告知用户稍后重试，**不要换写法反复尝试**。",
        }.get(err_kind, "如实告知用户解析失败的原因")
        return {
            "success": False,
            "operator": "resolve_place",
            "error_kind": err_kind,
            "error": result.get("error", f"未能解析地名：{place}"),
            "hint": hint + " 请如实告知用户，**切勿自行猜测坐标**。",
        }
    return {
        "success": True,
        "operator": "resolve_place",
        "result": {
            "name": result["name"],
            "lat": result["lat"],
            "lon": result["lon"],
            "display_name": result["display_name"],
        },
        "source": result["source"],
        "explanation": (
            f"地名『{place}』解析为 ({result['lat']}, {result['lon']})，"
            f"来源：{'本地词典' if result['source'] == 'local' else 'Nominatim 在线搜索（已写入本地词典）'}"
        ),
    }


register(
    "resolve_place",
    _resolve_place,
    "将地名（城市、地标等中文或拼音名）解析为经纬度坐标。"
    "本地词典命中则离线返回；未命中则在线搜索并写回词典。"
    "当用户只给了地名而非坐标时，先调用本算子获取坐标。",
    {
        "type": "object",
        "properties": {
            "place": {"type": "string", "description": "地名，如 '上海'、'杭州'、'Tokyo'、'黄山'"},
            "use_online": {"type": "boolean", "description": "本地未命中时是否允许在线搜索（默认 true）"},
        },
        "required": ["place"],
    },
)


def get_all_schemas() -> list[dict]:
    """返回所有算子的 OpenAI function schema 列表，用于 LLM tools 参数。"""
    return [schema for _, schema in _OPERATOR_REGISTRY.values()]


def list_operator_names() -> list[str]:
    """返回所有已注册算子的名称列表。"""
    return list(_OPERATOR_REGISTRY.keys())


def get_operator_func(name: str):
    """获取算子对应的 Python 函数；不存在则返回 None。"""
    if name not in _OPERATOR_REGISTRY:
        return None
    return _OPERATOR_REGISTRY[name][0]


def call_operator(name: str, **kwargs) -> dict:
    """按名称和参数调用算子，返回结果 dict。

    若算子名未知或参数错误，返回 success=False 的错误 dict，
    而非抛出异常，便于 LLM 在工具调用失败时获得可读反馈。
    """
    if name not in _OPERATOR_REGISTRY:
        return {"success": False, "error": f"未知算子: {name}"}
    func, _ = _OPERATOR_REGISTRY[name]
    try:
        return func(**_coerce_args(kwargs))
    except Exception as e:
        return {"success": False, "error": f"算子 {name} 执行出错: {e}"}
