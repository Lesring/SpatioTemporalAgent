"""地名 → 坐标 解析器（Gazetteer）。

三层解析策略（由快到慢）：
1. 本地地名词典（data/places.json）命中 → 离线、最快
2. 未命中 → 调用 Nominatim（OpenStreetMap，免费、无需 Key）在线地理编码
3. 在线解析成功 → **写回本地词典**，下次直接命中（自更新数据库）

在线搜索遵守 Nominatim 使用条款：请求间隔 ≥ 1.1s，遇 429 指数退避重试。

这样用户输入自然语言时只需说"上海到北京"，无需手打经纬度。
"""

import json
import os
import time
from typing import Optional

import httpx

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
)
_PLACES_PATH = os.path.join(_DATA_DIR, "places.json")

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {
    # Nominatim 使用条款要求提供 User-Agent。
    # 简洁的项目名/版本号格式经实测最不易被 OSMF 拒绝（避免 contact= 等异常字段）。
    "User-Agent": "spatio-temporal-agent/1.0"
}
# Nominatim 免费接口限速：至少 1 请求/秒。两次请求间留 1.1s 余量。
_MIN_REQUEST_INTERVAL_S = 1.1
_MAX_RETRIES = 3          # 429 / 网络异常最大重试次数
_RETRY_BACKOFF_BASE_S = 2.0  # 退避基数（秒）：2s, 4s, 8s

class GeocodeErrorKind:
    """区分在线解析失败的原因，避免把"限速/网络故障"误报成"查无此地"。"""
    NOT_FOUND = "not_found"       # Nominatim 明确返回空结果
    RATE_LIMITED = "rate_limited"  # 429，重试后仍被限
    BLOCKED = "blocked"           # 403，本机 IP / User-Agent 被 Nominatim 拒绝
    NETWORK = "network_error"     # 连接/超时/5xx/非 JSON 等网络层错误
    OK = "ok"


# 模块级：记录上次 Nominatim 请求时间，确保调用间隔满足限速
_last_nominatim_call_ts = 0.0


class Geocoder:
    """地名解析器，带本地词典 + Nominatim 在线兜底 + 写回。"""

    def __init__(self, places_path: str = _PLACES_PATH):
        self.places_path = places_path
        self._records: list[dict] = []
        self._index: dict[str, dict] = {}  # 小写名/别名 -> record
        self._load()
        self._last_online_hit: Optional[dict] = None

    # ── 加载 / 写回 ──────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self.places_path):
            self._records = []
            self._index = {}
            return
        with open(self.places_path, encoding="utf-8") as f:
            data = json.load(f)
        self._records = data.get("places", [])
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {}
        for rec in self._records:
            self._index[rec["name"].lower()] = rec
            for alias in rec.get("aliases", []):
                self._index[alias.lower()] = rec

    def save(self) -> None:
        """把当前词典（含在线回填的新条目）写回磁盘。"""
        os.makedirs(os.path.dirname(self.places_path), exist_ok=True)
        with open(self.places_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 用内存中的 records 覆盖 places（保留 _meta）
        data["places"] = self._records
        data.setdefault("_meta", {})
        data["_meta"]["updated_at"] = time.strftime("%Y-%m-%d")
        data["_meta"]["note"] = "地名词典。Nominatim 在线解析成功后会回填此文件。"
        with open(self.places_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 解析 ─────────────────────────────────────────────────
    def lookup_local(self, name: str) -> Optional[dict]:
        """仅查本地词典（离线）。返回 record 或 None。"""
        return self._index.get(name.strip().lower())

    def geocode(
        self, name: str, use_online: bool = True, write_back: bool = True
    ) -> dict:
        """解析地名为坐标。

        返回 dict：
        - 成功：{"success": True, "name", "lat", "lon", "display_name", "source"}
        - 失败：{"success": False, "name", "error_kind", "error"}
          error_kind ∈ GeocodeErrorKind
        """
        rec = self.lookup_local(name)
        if rec:
            return {
                "success": True,
                "name": rec["name"],
                "lat": rec["lat"],
                "lon": rec["lon"],
                "display_name": rec["name"],
                "source": "local",
            }

        if not use_online:
            return {
                "success": False,
                "name": name,
                "error_kind": GeocodeErrorKind.NOT_FOUND,
                "error": f"未在地名词典中找到：{name}",
            }

        online, err_kind = self._nominatim_search(name)
        if online is None:
            msg = {
                GeocodeErrorKind.NOT_FOUND: f"未找到地名『{name}』：本地词典与在线搜索均未命中（该地名确实无法解析）",
                GeocodeErrorKind.RATE_LIMITED: f"在线搜索被限速（429），暂时无法解析『{name}』，请稍后重试",
                GeocodeErrorKind.BLOCKED: f"Nominatim 拒绝本机访问（403），在线搜索不可用，无法解析『{name}』；可改用本地词典已收录地名或直接提供经纬度",
                GeocodeErrorKind.NETWORK: f"在线搜索网络异常，暂时无法解析『{name}』，请稍后重试",
            }.get(err_kind, f"在线搜索失败：{name}")
            return {
                "success": False,
                "name": name,
                "error_kind": err_kind,
                "error": msg,
            }

        self._last_online_hit = online
        if write_back:
            self._upsert_from_online(online)
            try:
                self.save()
            except Exception:
                # 写盘失败不影响本次解析
                pass
        return {
            "success": True,
            "name": online["name"],
            "lat": online["lat"],
            "lon": online["lon"],
            "display_name": online["display_name"],
            "source": online["source"],
        }

    # ── Nominatim 在线搜索 ───────────────────────────────────
    def _nominatim_search(
        self, name: str
    ) -> tuple[Optional[dict], str]:
        """调用 Nominatim，遵守限速并区分失败原因。

        返回 (记录或 None, 错误类型)。错误类型取自 GeocodeErrorKind。
        """
        global _last_nominatim_call_ts
        last_error = GeocodeErrorKind.NETWORK

        for attempt in range(_MAX_RETRIES):
            # 限速：确保距上次请求 ≥ 1.1s
            wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_nominatim_call_ts)
            if wait > 0:
                time.sleep(wait)

            _last_nominatim_call_ts = time.monotonic()
            params = {
                "q": name,
                "format": "json",
                "limit": 1,
                "addressdetails": 0,
            }
            try:
                resp = httpx.get(
                    _NOMINATIM_URL,
                    params=params,
                    headers=_NOMINATIM_HEADERS,
                    timeout=15,
                )
                if resp.status_code == 429:
                    last_error = GeocodeErrorKind.RATE_LIMITED
                    backoff = _RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                if resp.status_code == 403:
                    # Nominatim 持续拒绝本机 IP / User-Agent，重试无效
                    last_error = GeocodeErrorKind.BLOCKED
                    break
                if resp.status_code >= 500:
                    last_error = GeocodeErrorKind.NETWORK
                    backoff = _RETRY_BACKOFF_BASE_S * (2 ** attempt)
                    time.sleep(backoff)
                    continue

                resp.raise_for_status()
                results = resp.json()
            except Exception:
                last_error = GeocodeErrorKind.NETWORK
                backoff = _RETRY_BACKOFF_BASE_S * (2 ** attempt)
                time.sleep(backoff)
                continue

            # 明确返回空结果 = 查无此地（不是网络故障）
            if not results:
                return None, GeocodeErrorKind.NOT_FOUND

            r = results[0]
            return {
                "name": r.get("name") or name,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "display_name": r.get("display_name", name),
                "source": "nominatim",
                "osm_type": r.get("type"),
                "place_id": r.get("place_id"),
            }, GeocodeErrorKind.OK

        # 所有重试均失败
        return None, last_error

    def _upsert_from_online(self, rec: dict) -> None:
        """把在线结果并入本地词典（若不存在），并注册别名。"""
        key = rec["name"].strip().lower()
        if key in self._index:
            return
        entry = {
            "name": rec["name"],
            "lat": rec["lat"],
            "lon": rec["lon"],
            "type": "city",
            "aliases": [],
            "_origin": "nominatim",
        }
        self._records.append(entry)
        self._index[key] = entry
        for alias in [rec.get("display_name"), rec.get("osm_type")]:
            if alias and alias.lower() not in self._index:
                entry["aliases"].append(alias)
                self._index[alias.lower()] = entry
