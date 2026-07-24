# tools/filing_guide.py申报表填报指导
from langchain.tools import tool
from retriever import get_retriever
from model import Chat_model
from logger_config import setup_logger

logger = setup_logger("filing_guide",log_file="filing_guide.log")

@tool
def filing_guide_tool(form_name: str, business_details: str = "") -> str:
    """
    提供增值税申报表（如附表一、附表二）的填报指导。
    输入：申报表名称（如“附表二”）、业务详情（可选）
    输出：填报规则、注意事项及示例。
    """
    logger.info(f"申报表填报指导工具被调用，表单: {form_name}, 详情: {business_details}")
    try:
        retriever = get_retriever()
        # 构建查询
        query = f"{form_name} 填报规则 增值税申报表"
        results, _ = retriever.retrieve(query, top_k=5)
        if not results:
            return f"未找到关于 {form_name} 的填报规则，请确认表单名称是否正确。"

        context = "\n".join([r["text"] for r in results])
        # 用LLM生成结构化指导
        prompt = f"""
        你是一个税务专家，请根据以下法规内容，为用户提供关于 {form_name} 的填报指导。
        用户业务详情（如果有）：{business_details}

        法规参考：
        {context}

        请按以下格式回答：
        1. 填报依据
        2. 关键栏位说明
        3. 注意事项
        4. 常见错误示例
        """
        response = Chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"填报指导工具异常: {e}", exc_info=True)
        return f"获取填报指导失败：{str(e)}"