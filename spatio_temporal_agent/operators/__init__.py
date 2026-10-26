"""时空地理计算基本算子模块。

本模块提供一组基础地理信息（GIS）算子，供智能体调用执行空间/时间计算任务。
算子设计原则：
- 每个算子是一个纯函数，有明确的参数 schema（用于 LLM function calling）
- 计算结果以 JSON 可序列化的 dict 返回
"""

from .geo_operators import (
    haversine_distance,
    bearing,
    destination_point,
    point_in_polygon,
    polygon_area,
    bounding_box_overlap,
    buffer_circle,
    nearest_point,
    polygon_centroid,
    line_length,
    angle_between,
    temporal_distance,
    timezone_convert,
    time_range_filter,
    time_series_resample,
)

__all__ = [
    "haversine_distance",
    "bearing",
    "destination_point",
    "point_in_polygon",
    "polygon_area",
    "bounding_box_overlap",
    "buffer_circle",
    "nearest_point",
    "polygon_centroid",
    "line_length",
    "angle_between",
    "temporal_distance",
    "timezone_convert",
    "time_range_filter",
    "time_series_resample",
]
