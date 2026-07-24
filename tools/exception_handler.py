# tools/exception_handler.py异常处理建议
from langchain.tools import tool
from retriever import get_retriever
from model import Chat_model
from logger_config import setup_logger

logger = setup_logger("exception_handler",log_file="exception_handler.log")

@tool
def exception_handler_tool(exception_code: str, description: str = "") -> str:
    """
    根据申报异常代码（如 A05014）或异常描述，提供处理建议和操作步骤。
    输入：异常代码（字符串）或描述
    输出：异常原因、自查方法、处理流程、联系方式。
    """
    logger.info(f"异常处理工具被调用，代码: {exception_code}, 描述: {description}")
    try:
        retriever = get_retriever()
        query = f"{exception_code} 增值税申报异常 处理"
        if description:
            query += f" {description}"
        results, _ = retriever.retrieve(query, top_k=5)
        context = "\n".join([r["text"] for r in results]) if results else "未找到匹配异常信息。"

        # 若知识库无直接匹配，用LLM结合常见规则生成建议
        prompt = f"""
        你是税务申报异常处理专家。
        异常代码：{exception_code}
        用户描述：{description}

        参考法规与指引（可能不直接相关）：
        {context}

        请给出以下内容：
        1. 可能的原因
        2. 自查步骤（检查哪些数据）
        3. 处理流程（如何更正或联系税务机关）
        4. 若无法解决，建议联系主管税务机关。
        """
        response = Chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"异常处理工具异常: {e}", exc_info=True)
        return f"获取异常处理建议失败：{str(e)}"