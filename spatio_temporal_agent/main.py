"""主程序入口。

用法：
    python -m spatio_temporal_agent
    或
    python spatio_temporal_agent/main.py

交互模式：输入自然语言地理计算问题，智能体自动调用算子并返回结果。
"""

import json
import sys
from pathlib import Path

from .agent import LLMClient, TaskExecutor
from .operators.registry import get_all_schemas


def main() -> None:
    # ── 初始化 LLM 客户端 ───────────────────────────────────
    # 优先从 config.json 读取，否则从环境变量读取
    config_path = Path(__file__).parent / "config.json"
    try:
        llm = LLMClient.from_config_file(str(config_path))
    except ValueError as e:
        print(f"[配置错误] {e}")
        print("请在项目目录创建 config.json：")
        print(json.dumps({
            "api_base": "https://api.openai.com/v1",
            "api_key": "<你的API_KEY>",
            "model": "gpt-4o",
        }, indent=2, ensure_ascii=False))
        sys.exit(1)

    executor = TaskExecutor(llm=llm)

    print("=" * 60)
    print("  时空地理计算智能体（B / TaskExecutor）")
    print("  支持算子：", ", ".join(s["function"]["name"] for s in get_all_schemas()))
    print("  HTTP API：python -m spatio_temporal_agent.api.server")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n请输入地理计算问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        result = executor.run("cli", user_input)
        data = result.to_dict()

        print("\n── 执行结果 ──")
        print(f"final_status: {data['process']['final_status']}  success: {data['success']}")
        for step in data["process"]["steps"]:
            if step["type"] == "tool_call":
                print(f"  • [{step['status']}] {step['name']}({step['args']})")
                print(f"    → {step.get('message') or step.get('result')}")
        print(f"\n回答：{data['answer']}")
        if data.get("error"):
            print(f"错误：{data['error']}")


if __name__ == "__main__":
    main()
