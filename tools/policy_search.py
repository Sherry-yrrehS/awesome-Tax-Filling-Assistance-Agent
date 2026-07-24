# tools/policy_search.py政策检索工具
from langchain.tools import tool
from retriever import get_retriever
from logger_config import setup_logger

logger = setup_logger("policy_search", log_file="policy_search.log")


@tool
def policy_search_tool(query: str) -> str:
    """
    根据用户提出的税务问题，检索增值税相关法规条款并返回原文引用。
    适用场景：用户询问具体政策规定（如税率、征收范围、优惠条件等）。
    输入：用户问题（字符串）
    输出：包含条款原文和来源的文本，若未找到则返回提示。
    """
    logger.info(f"政策检索工具被调用，查询: {query}")
    try:
        retriever = get_retriever()
        results, confidence = retriever.retrieve(query, top_k=5)
        if not results:
            return "未找到与您问题直接相关的法规条款，建议您提供更具体的信息或咨询税务机关。"

        # 格式化输出
        output_lines = []
        for idx, item in enumerate(results, 1):
            text = item.get("text", "")
            source = item.get("source", "未知来源")
            score = item.get("score", 0.0)
            output_lines.append(f"【片段 {idx}】（来源：{source}，相关度：{score:.2f}）\n{text}\n")

        # 若置信度低于阈值，附加提示
        if confidence < 0.5:
            output_lines.append("\n⚠️ 当前检索结果置信度较低，建议您核对原文或咨询税务机关。")

        return "\n".join(output_lines)
    except Exception as e:
        logger.error(f"政策检索工具异常: {e}", exc_info=True)
        return f"检索失败，请稍后重试。错误信息：{str(e)}"