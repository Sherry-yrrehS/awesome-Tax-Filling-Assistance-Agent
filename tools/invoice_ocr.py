# tools/invoice_ocr.py 发票OCR + 抵扣判定
import os
from langchain.tools import tool
from logger_config import setup_logger

logger = setup_logger("invoice_ocr",log_file="invoice_ocr.log")

# 模拟OCR提取函数（实际可替换为PaddleOCR或GPT-4V）
def fake_ocr(image_path: str) -> dict:
    """模拟从发票图片提取结构化信息（演示用）"""
    # 实际应调用 OCR 服务
    return {
        "invoice_code": "1234567890",
        "invoice_number": "12345678",
        "buyer_tax_id": "91110101000000000X",
        "seller_tax_id": "91110101000000000Y",
        "amount": 10000.00,
        "tax_amount": 1300.00,
        "date": "2026-01-01"
    }

@tool
def invoice_ocr_tool(image_path: str) -> str:
    """
    上传增值税发票图片（路径），自动识别票面信息并判断该发票的进项税额是否可以抵扣。
    输入：图片文件路径（字符串）
    输出：抵扣判定结果及依据（若支持抵扣，返回法规引用；否则说明原因）
    """
    logger.info(f"发票OCR工具被调用，图片路径: {image_path}")
    if not os.path.exists(image_path):
        return f"文件不存在: {image_path}"

    try:
        # 1. OCR 提取信息（此处为模拟）
        info = fake_ocr(image_path)
        logger.info(f"OCR提取结果: {info}")

        # 2. 简单抵扣逻辑（实际应根据发票类型、税率、业务性质等综合判断）
        # 这里仅作演示：假设金额>0且税率合规即为可抵扣
        if info.get("tax_amount", 0) > 0:
            # 检索相关法规（例如“不得抵扣情形”）
            from retriever import get_retriever
            retriever = get_retriever()
            query = "增值税进项税额不得抵扣的情形"
            results, _ = retriever.retrieve(query, top_k=3)
            context = "\n".join([r["text"] for r in results]) if results else "未检索到具体条款。"

            # 3. 用LLM生成最终判定（简单模拟）
            from model import Chat_model
            prompt = f"""
            根据以下发票信息：
            金额：{info['amount']} 元，税额：{info['tax_amount']} 元
            购买方税号：{info['buyer_tax_id']}
            销售方税号：{info['seller_tax_id']}

            参考法规：
            {context}

            请判断该发票进项税额是否允许抵扣，并给出简要理由。
            """
            response = Chat_model.invoke(prompt)
            return f"【发票信息】\n{info}\n\n【判定结果】\n{response.content}"
        else:
            return "该发票税额为0，无法抵扣进项税额。"
    except Exception as e:
        logger.error(f"发票处理异常: {e}", exc_info=True)
        return f"发票处理失败，请确认图片有效。错误：{str(e)}"