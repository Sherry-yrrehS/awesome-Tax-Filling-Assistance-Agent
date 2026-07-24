# agent_graph.py - LangGraph Agent 定义（ReAct 模式）
from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool

from model import Chat_model
from tools import (
    policy_search_tool,
    invoice_ocr_tool,
    filing_guide_tool,
    exception_handler_tool,
    escalate_tool,
)
from logger_config import setup_logger

logger = setup_logger("agent_graph", log_file="agent_graph.log")

# ---------------------- 1. 定义系统提示词 ----------------------
SYSTEM_PROMPT = """你是一位专业的增值税智能税务助手，专门为中小企业财务人员和办税员提供政策咨询、发票抵扣判定、申报表填报指导和异常处理建议。

你的知识库基于《中华人民共和国增值税法》及2026年最新政策。你必须严格依据检索到的法规条款回答，并注明来源（文件名称）。

可用工具：
1. policy_search_tool(query)：检索增值税相关法规条款，返回原文引用。
2. invoice_ocr_tool(image_path)：识别发票图片并判断进项税额能否抵扣。
3. filing_guide_tool(form_name, business_details)：提供申报表填报指导。
4. exception_handler_tool(exception_code, description)：提供异常处理建议。
5. escalate_tool(user_query, reason)：当无法回答时，引导用户咨询税务机关。

工作流程：
- 对于一般政策咨询，直接调用 policy_search_tool。
- 如果用户问题涉及发票抵扣，调用 invoice_ocr_tool。
- 如果用户明确提到申报表名称（如“附表二”），调用 filing_guide_tool。
- 如果用户提供异常代码或描述，调用 exception_handler_tool。
- 如果工具返回结果置信度低或未找到相关依据，应调用 escalate_tool 并给出解释。

回答要求：
- 引用法规原文时，必须标注来源（例如：根据《增值税法》第十条…）。
- 若涉及具体计算，给出计算公式和示例。
- 最后提醒用户：本回答仅供参考，具体以税务机关执行为准。
"""

# ---------------------- 2. 创建工具列表 ----------------------
tools = [
    policy_search_tool,
    invoice_ocr_tool,
    filing_guide_tool,
    exception_handler_tool,
    escalate_tool,
]

# 将 @tool 装饰的函数转换为 LangChain Tool 对象（create_react_agent 要求）
from langchain_core.tools import tool as langchain_tool
# 如果 tools 已经是 Tool 对象，则直接使用；否则转换
# 注意：@tool 装饰器默认会生成符合要求的 Tool 对象，所以可以直接传入

# ---------------------- 3. 创建记忆（支持多轮对话）---------------------
from memory import checkpointer

# ---------------------- 4. 创建 ReAct Agent ----------------------
agent = create_agent(
    model=Chat_model,               # 大模型实例
    tools=tools,                    # 工具列表
    system_prompt=SYSTEM_PROMPT,   # 系统提示词（相当于 system message）
    checkpointer=checkpointer,            # 记忆存储，实现多轮对话
)

logger.info("Agent 创建成功，已加载 %d 个工具", len(tools))

# ---------------------- 5. 包装调用函数（可选）---------------------
def run_agent(query: str, thread_id: str = "default") -> str:
    """
    外部调用入口，将用户问题转换为消息格式并调用 agent。
    :param query: 用户问题（字符串）
    :param thread_id: 会话标识，用于区分不同用户/对话（默认 "default"）
    :return: Agent 最终回答（字符串）
    """
    logger.info(f"开始处理问题（thread_id={thread_id}）：{query}")
    try:
        # 调用 agent，输入为消息列表（HumanMessage）
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        # 提取最后一条 AI 消息内容
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            answer = final_message.content
        else:
            # 如果最后一条不是 AI 消息，尝试获取 content
            answer = str(final_message.content) if hasattr(final_message, "content") else str(final_message)
        logger.info(f"回答生成完成，长度：{len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"Agent 调用失败: {e}", exc_info=True)
        return f"系统处理出错，请稍后重试。错误：{str(e)}"

# 若需要清空记忆，可提供重置函数（可选）
def reset_memory(thread_id: str = "default"):
    """清除指定会话的记忆"""
    # MemorySaver 没有直接删除方法，但可以通过重新创建覆盖
    # 简单处理：将对应 thread_id 的状态置空
    # 实际使用中，可以维护一个字典，但 MemorySaver 内部存储不可直接操作
    # 我们建议为每个新会话分配新 thread_id
    logger.warning(f"重置记忆（thread_id={thread_id}）需要重新创建 agent 实例，暂不支持")