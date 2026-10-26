"""领域知识库模块。

基于 OpenAI 嵌入接口构建语义检索增强的领域知识库，
同时提供离线关键词匹配 fallback，确保没有 API Key 时也能运行演示。

知识库内容由 data/knowledge_items.json 加载。
"""

import json
import os
import re
from typing import Optional

# 知识库数据路径（相对于项目根目录）
_KNOWLEDGE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "knowledge_items.json",
)


class KnowledgeBase:
    """基于关键词匹配的轻量领域知识库（离线 fallback）。

    支持后续扩展为向量检索（RAG）。
    """

    def __init__(self, data_path: str = _KNOWLEDGE_DATA_PATH):
        self._items: list[dict] = []
        self._load(data_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            self._items = json.load(f)

    @property
    def all_items(self) -> list[dict]:
        return self._items

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """根据查询关键词返回最相关的前 top_k 条知识条目。

        使用简单的词频匹配评分，无需外部依赖。
        """
        if not self._items:
            return []

        query_tokens = set(self._tokenize(query))
        scored = []
        for item in self._items:
            item_text = " ".join([
                item.get("title", ""),
                item.get("description", ""),
                " ".join(item.get("keywords", [])),
            ])
            item_tokens = set(self._tokenize(item_text))
            # 重叠关键词越多，得分越高
            score = len(query_tokens & item_tokens)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def build_context(self, query: str, top_k: int = 3) -> str:
        """为 LLM 提示词构建检索增强的上下文块。"""
        results = self.search(query, top_k=top_k)
        if not results:
            return "（暂无相关领域知识）"
        lines = []
        for i, item in enumerate(results, 1):
            lines.append(
                f"[{i}] {item['title']}\n"
                f"    描述: {item.get('description', '')}\n"
                f"    关键词: {', '.join(item.get('keywords', []))}"
            )
        return "\n".join(lines)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词：按标点/空格切分，保留中英文。"""
        # 去除标点，按字符级提取中英文 token
        text = text.lower()
        tokens = re.split(r"[^\w\u4e00-\u9fff]+", text)
        # 对中文做 2-gram 切分
        result = []
        for t in tokens:
            if re.match(r"^[\u4e00-\u9fff]+$", t):
                # 2-gram
                for i in range(len(t) - 1):
                    result.append(t[i : i + 2])
                if len(t) == 1:
                    result.append(t)
            else:
                result.append(t)
        return result
