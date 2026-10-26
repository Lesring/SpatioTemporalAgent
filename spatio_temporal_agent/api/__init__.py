"""B 层 HTTP API（供 C 前端 / 批量评测对接）。"""

from .server import create_app

__all__ = ["create_app"]
