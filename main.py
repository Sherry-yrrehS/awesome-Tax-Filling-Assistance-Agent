# main.py - 系统入口（FastAPI + 命令行双模式）
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from logger_config import setup_logger
from agent_graph import run_agent
from memory import checkpointer  # 确保记忆初始化

# 加载环境变量
load_dotenv(override=True)

# 配置日志
logger = setup_logger("main", log_file="main.log")

# ---------------------- FastAPI 应用 ----------------------
app = FastAPI(
    title="增值税智能税务助手 API",
    description="面向中小企业财务人员的增值税政策咨询、发票抵扣判定、申报指导、异常处理",
    version="1.0.0"
)

# ---------------------- 工具函数 ----------------------
def get_or_create_thread_id(user_id: Optional[str] = None) -> str:
    """
    获取或生成会话 ID（thread_id）
    - 若提供 user_id，则使用它作为 thread_id（便于跨会话记忆）
    - 否则生成随机 UUID
    """
    if user_id:
        return f"user_{user_id}"
    return str(uuid.uuid4())

# ---------------------- API 端点 ----------------------
@app.post("/chat", response_model=dict)
async def chat(
    query: str = Form(..., description="用户问题"),
    user_id: Optional[str] = Form(None, description="用户标识（可选）")
):
    """
    文本问答接口
    - 用户提交问题，Agent 自动调用工具检索并回答
    - 支持多轮对话（通过 user_id 维持会话记忆）
    """
    logger.info(f"收到聊天请求，user_id={user_id}, query={query}")
    try:
        thread_id = get_or_create_thread_id(user_id)
        answer = run_agent(query, thread_id=thread_id)
        return JSONResponse({
            "success": True,
            "answer": answer,
            "thread_id": thread_id
        })
    except Exception as e:
        logger.error(f"聊天处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.post("/upload")
async def upload_invoice(
    file: UploadFile = File(..., description="发票图片文件"),
    user_id: Optional[str] = Form(None)
):
    """
    上传发票图片，自动识别并判断抵扣
    - 保存图片到临时目录，调用 invoice_ocr_tool 处理
    """
    logger.info(f"收到图片上传，user_id={user_id}, filename={file.filename}")
    # 检查文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    try:
        # 保存临时文件
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 构造问题，调用 Agent（直接调用工具）
        from tools import invoice_ocr_tool
        result = invoice_ocr_tool(str(file_path))

        # 清理临时文件（可选）
        # file_path.unlink(missing_ok=True)

        return JSONResponse({
            "success": True,
            "result": result,
            "thread_id": get_or_create_thread_id(user_id)
        })
    except Exception as e:
        logger.error(f"图片处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "memory": "SQLite" if checkpointer else "memory"}

# ---------------------- 命令行交互模式 ----------------------
def interactive_mode():
    """命令行交互式问答（支持多轮对话）"""
    print("=" * 60)
    print("增值税智能税务助手 (命令行版)")
    print("输入 'exit' 或 'quit' 退出，输入 'clear' 重置会话")
    print("=" * 60)
    thread_id = "cli_session"  # 固定会话 ID，保持连续对话

    while True:
        try:
            query = input("\n🧾 您的问题: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("感谢使用，再见！")
                break
            if query.lower() == "clear":
                # 重置记忆（对 SQLite 暂不支持，此处提示）
                print("⚠️ 目前不支持清空记忆，可重启程序或删除 checkpoints.db")
                continue

            print("🤖 思考中...")
            answer = run_agent(query, thread_id=thread_id)
            print("\n" + "=" * 60)
            print(answer)
            print("=" * 60)
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            logger.error(f"交互模式异常: {e}", exc_info=True)
            print(f"⚠️ 出错了: {e}")

# ---------------------- 主入口 ----------------------
def main():
    """根据命令行参数决定启动模式"""
    if len(sys.argv) > 1:
        # 如果有参数，尝试作为问题直接回答（单次问答）
        query = " ".join(sys.argv[1:])
        print(f"🧾 问题: {query}")
        answer = run_agent(query, thread_id="single_shot")
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
    else:
        # 无参数则启动交互模式
        interactive_mode()

if __name__ == "__main__":
    # 如果直接运行 main.py，且无参数，启动交互模式
    # 若需要启动 FastAPI 服务，请使用：uvicorn main:app --reload
    if len(sys.argv) == 1:
        # 检查是否想启动 API 服务（可通过环境变量或参数控制）
        # 这里简单处理：无参数启动交互模式
        main()
    else:
        # 有参数时，尝试作为单次问答
        main()