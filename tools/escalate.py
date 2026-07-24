# tools/escalate.py 转人工/引导至税务机关
from langchain.tools import tool
from logger_config import setup_logger

logger = setup_logger("escalate")

@tool
def escalate_tool(user_query: str, reason: str = "无法找到相关依据") -> str:
    """
    当 Agent 无法准确回答用户问题时，引导用户咨询税务机关或拨打 12366。
    输入：用户原始问题、失败原因（可选）
    输出：标准引导话术
    """
    logger.info(f"引导工具被调用，问题: {user_query}, 原因: {reason}")
    guide = f"""
    非常抱歉，我目前的知识库暂未能找到关于您问题的充分依据。
    您的问题是：{user_query}
    原因：{reason}

    为确保您获得准确权威的答复，建议您：
    1. 拨打全国税务服务热线 **12366** 进行人工咨询。
    2. 携带相关资料前往您的主管税务机关办税服务厅。
    3. 访问国家税务总局官网（www.chinatax.gov.cn）查询最新政策。

    我们正在不断完善知识库，感谢您的理解与支持。
    """
    return guide