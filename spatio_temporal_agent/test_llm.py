"""端到端测试：使用真实 LLM（Agnes-AI / OpenAI 兼容接口）验证智能体完整流程。

用法：
    python -m spatio_temporal_agent.test_llm
"""

from .agent import LLMClient, SpatioTemporalAgent


def main() -> None:
    llm = LLMClient.from_config_file(
        "spatio_temporal_agent/config.json"
    )
    print(f"使用模型: {llm.model} @ {llm.api_base}")
    agent = SpatioTemporalAgent(llm=llm)

    test_cases = [
        # 空间：纯地名，无需手打坐标（考察 resolve_place 自动编排）
        "计算上海到杭州的距离",
        "判断点(31.25, 121.45)是否在多边形[[31.2,121.4],[31.2,121.5],[31.3,121.5],[31.3,121.4]]内部",
        "以上海为圆心，生成半径5000米的缓冲区",
        # 时间：考察时间算子
        "2026-10-18 早上8点(北京时间)到 晚上8点(北京时间) 相隔多少小时",
        "把北京时间 2026-10-18 08:00 换算成伦敦时间",
        # 多步骤：先解析地名再算方位
        "从北京到成都的方位角是多少",
    ]

    for i, q in enumerate(test_cases, 1):
        print(f"\n{'='*60}\n测试 {i}: {q}\n{'='*60}")
        try:
            result = agent.ask(q)
            print(f"回答: {result['answer']}")
            if result["tool_calls_made"]:
                print("调用记录:")
                for tc in result["tool_calls_made"]:
                    print(f"  • {tc['operator']} → {tc['result'].get('explanation') or tc['result']}")
        except Exception as e:
            print(f"[错误] {type(e).__name__}: {e}")
            break


if __name__ == "__main__":
    main()
