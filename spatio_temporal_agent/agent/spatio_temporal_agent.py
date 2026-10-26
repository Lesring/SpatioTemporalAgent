"""时空地理计算智能体核心。

智能体流程：
1. 接收用户自然语言输入
2. 检索领域知识库，构建上下文
3. 调用 LLM（带 tools），让模型判断需要调用哪些算子
4. 执行算子，将结果返回给 LLM
5. LLM 组织最终自然语言回答
"""

import json
import re
from typing import Any, Optional

from ..knowledge_base import KnowledgeBase
from ..operators.registry import call_operator, get_all_schemas, list_operator_names
from .llm_client import LLMClient


def _strip_tool_call_artifacts(text: str) -> str:
    """清洗 LLM 误把工具调用混进最终回答的文本。

    agnes-2.5-flash 在任务复杂 / 轮次多时，有时会在 content 里吐出
    形如 "resolve_place(...)【参数】...【输出】..." 的工具调用残片，
    而不是自然语言。此函数把这些残片识别并移除，保留真正的回答。
    若清洗后为空，则回退为提示文案。
    """
    if not text or not text.strip():
        return text
    names = list_operator_names()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # 跳过明显是工具调用 / 参数 / 输出标记的行
        if not stripped:
            continue
        if re.match(r"^[•\-\*\s]*\[?[A-Za-z_]+\(", stripped):
            continue
        if stripped.startswith(("【参数】", "【输出】", "【结果】")):
            continue
        if any(n in stripped for n in names) and (
            stripped.startswith(n) or f"({n}" in stripped or f"{n}(" in stripped
        ):
            # 该行像工具调用残片，跳过
            if len(stripped) < 120 and any(
                k in stripped for k in ("resolve_place", "haversine", "bearing", "buffer", "【")
            ):
                continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return "（计算未完成：LLM 输出中未包含有效的自然语言总结，请简化问题或分步提问。）"
    return cleaned


class SpatioTemporalAgent:
    """时空地理计算智能体。"""

    def __init__(
        self,
        llm: LLMClient,
        kb: Optional[KnowledgeBase] = None,
        max_tool_rounds: int = 8,
        max_tool_calls: int = 20,
    ):
        self.llm = llm
        self.kb = kb or KnowledgeBase()
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls = max_tool_calls  # 单次提问的总调用次数上限，防盲猜失控
        self._tools = get_all_schemas()

    def ask(self, user_input: str) -> dict:
        """处理用户输入，返回完整执行结果。

        返回 dict：
        - answer: 最终自然语言回答
        - tool_calls_made: 执行过的算子调用记录（含 elapsed_ms）
        - kb_context: 检索到的知识上下文文本
        - kb_hits: 检索命中的知识条目列表
        """
        import time as _time

        kb_hits = self.kb.search(user_input)
        kb_context = self.kb.build_context(user_input)

        system_prompt = self._build_system_prompt(kb_context)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        tool_calls_made: list[dict] = []

        for _ in range(self.max_tool_rounds):
            resp = self.llm.chat(messages, tools=self._tools)
            msg = resp["message"]
            tool_calls = resp.get("tool_calls") or []

            if not tool_calls:
                answer = _strip_tool_call_artifacts(msg.get("content") or "")
                return {
                    "answer": answer,
                    "tool_calls_made": tool_calls_made,
                    "kb_context": kb_context,
                    "kb_hits": kb_hits,
                }

            messages.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                if len(tool_calls_made) >= self.max_tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({
                            "success": False,
                            "error": "已达本次提问的最大算子调用次数，请立即停止调用工具，"
                                      "基于已获得的工具结果输出自然语言总结，并注明信息可能不完整。",
                        }, ensure_ascii=False),
                    })
                    continue

                func_info = tc["function"]
                name = func_info["name"]
                try:
                    args = json.loads(func_info["arguments"])
                except json.JSONDecodeError:
                    args = {}

                t0 = _time.perf_counter()
                result = call_operator(name, **args)
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                tool_calls_made.append({
                    "operator": name,
                    "args": args,
                    "result": result,
                    "elapsed_ms": round(elapsed_ms, 2),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

        messages.append({
            "role": "user",
            "content": (
                "你现在不要再调用任何工具，仅根据上面已经完成的工具调用结果，"
                "用简洁的自然语言总结最终答案。"
                "不要输出工具调用、JSON、代码块或 '【参数】' 之类的字样；"
                "如果某个地名未能解析，就如实说明，不要猜测坐标。"
            ),
        })
        resp = self.llm.chat(messages)
        answer = resp["message"].get("content") or "（未能生成最终回答）"
        answer = _strip_tool_call_artifacts(answer)

        return {
            "answer": answer,
            "tool_calls_made": tool_calls_made,
            "kb_context": kb_context,
            "kb_hits": kb_hits,
        }

    def _build_system_prompt(self, kb_context: str) -> str:
        return f"""你是一名时空地理计算智能体。
你的任务是：理解用户的自然语言地理/时间计算请求，
通过调用基本算子完成计算，并以自然语言返回结果。

可用算子：
{json.dumps([s['function']['name'] for s in self._tools], ensure_ascii=False)}

领域知识库检索结果（供参考，辅助判断算子选择和参数说明）：
{kb_context}

执行规则：
1. 若用户只给了地名（如"上海""北京""黄山"）而没有坐标，
   必须先调用 resolve_place 把地名解析为坐标，再用坐标调用距离/方位/缓冲区等算子；
   一个请求可能涉及多次算子调用（先解析地名，再做计算），请按需连续调用。
   **同名地名只需调用一次 resolve_place**，不要反复用不同写法重试；
   若失败且 error_kind 为 rate_limited 或 network_error，属暂时性故障，直接告知用户稍后重试，**不要更换写法反复尝试**。
2. 若意图是时间相关（两个时间间隔、时区换算、时间段过滤、时序重采样），
   选择对应的时间算子；时间戳尽量用 ISO 8601 格式。
3. 若参数缺失，向用户追问。
4. 若某算子返回 success=False，说明失败原因并给出修正建议。
   **严禁在算子失败后用你自己的记忆/常识补坐标或数值**——
   地名解析失败就如实告知用户解析失败及原因，不得自行猜测经纬度后继续计算。
5. 对于"附近有哪些 X"这类邻近/列举问题：
   本组算子不具备"POI 检索"能力，无法穷举某点周边所有公园/地标。
   正确做法：最多列举 2-3 个你较为确定的候选地名，逐个 resolve_place + haversine_distance 验证是否在范围内，
   然后说明"以下为已验证候选，非完整列表"，并提示用户可补充更精确的地名。
   **不要一次盲猜 10+ 个候选名**，避免产生大量无用的算子调用。
6. 一旦收集到足够信息，**立即停止调用工具**，直接输出自然语言最终回答；
   不要在没有新信息的情况下继续追加算子调用。
7. 最终回答要简洁、准确，包含数值、单位和必要的坐标；
   **禁止在最终回答里输出工具调用、JSON、【参数】/【输出】或代码块**——回答必须是给用户的自然语言。
"""
