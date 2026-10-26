"""LLM 客户端封装，支持 OpenAI 兼容 API（OpenAI / DeepSeek / 通义 / Moonshot 等）。

配置通过环境变量或 config.json 读取：
- LLM_API_BASE: API 基础地址（默认 https://api.openai.com/v1）
- LLM_API_KEY:  API Key
- LLM_MODEL:    模型名（默认 gpt-4o）
"""

import json
import os
import time
from typing import Any, Optional

# 默认配置
_DEFAULTS = {
    "api_base": os.environ.get("LLM_API_BASE", "https://api.openai.com/v1"),
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "model": os.environ.get("LLM_MODEL", "gpt-4o"),
}


class LLMClient:
    """OpenAI 兼容接口客户端。

    使用 httpx 避免额外安装 openai 包；若环境有 openai 库也可直接用。
    支持 function calling（tools）接口。
    """

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "",
    ):
        self.api_base = (api_base or _DEFAULTS["api_base"]).rstrip("/")
        self.api_key = api_key or _DEFAULTS["api_key"]
        self.model = model or _DEFAULTS["model"]

        if not self.api_key:
            raise ValueError(
                "LLM API Key 未配置。请设置环境变量 LLM_API_KEY "
                "或传入 api_key 参数。"
            )

    @classmethod
    def from_config_file(cls, path: str = "config.json") -> "LLMClient":
        """从 config.json 加载配置。"""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            return cls(
                api_base=cfg.get("api_base", ""),
                api_key=cfg.get("api_key", ""),
                model=cfg.get("model", ""),
            )
        return cls()

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
    ) -> dict:
        """调用 LLM chat completions 接口，返回原始 response。

        返回 dict 包含：
        - message: assistant 消息 dict
        - tool_calls: 工具调用列表（若有）
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 简单重试 + 指数退避，应对 429 限流 / 5xx 瞬时错误 / 连接级网络异常
        import httpx

        transient_exceptions = (
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
        )

        max_retries = 5
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = httpx.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
            except transient_exceptions as exc:
                wait = min(2 ** attempt * 1.5, 30)
                last_exc = exc
                print(f"  [LLM] 网络异常 {type(exc).__name__}，{wait:.1f}s 后重试 "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                wait = retry_after if retry_after > 0 else min(2 ** attempt * 1.5, 30)
                last_exc = httpx.HTTPStatusError(
                    f"{resp.status_code}", request=resp.request, response=resp
                )
                print(f"  [LLM] 收到 {resp.status_code}，{wait:.1f}s 后重试 "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            break
        else:
            raise last_exc  # type: ignore[misc]

        data = resp.json()
        choice = data["choices"][0]["message"]
        return {
            "message": choice,
            "tool_calls": choice.get("tool_calls") or [],
            "raw": data,
        }
