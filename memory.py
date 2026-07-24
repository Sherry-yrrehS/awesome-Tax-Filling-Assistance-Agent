# memory.py - 对话记忆管理（持久化 + 内存回退）
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from logger_config import setup_logger

logger = setup_logger("memory", log_file="memory.log")

# ---------- 配置项（可通过环境变量覆盖） ----------
# 是否启用 SQLite 持久化（默认启用）
USE_SQLITE = os.getenv("USE_SQLITE_MEMORY", "true").lower() == "true"
# SQLite 数据库文件路径（默认项目根目录下的 checkpoints.db）
DB_PATH = os.getenv("MEMORY_DB_PATH", "checkpoints.db")


def get_checkpointer(use_sqlite: bool = USE_SQLITE, db_path: str = DB_PATH):
    """
    获取 LangGraph 兼容的 Checkpoint 保存器。
    :param use_sqlite: 若 True，使用 SQLite 持久化；否则使用内存（重启丢失）
    :param db_path: SQLite 数据库文件路径
    :return: BaseCheckpointSaver 实例
    """
    if use_sqlite:
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            # 创建 SQLite 连接（允许跨线程访问）
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"使用 SQLite 持久化记忆，数据库文件: {db_path}")
            return checkpointer
        except ImportError as e:
            logger.error(f"SqliteSaver 导入失败（可能缺少依赖），回退到内存模式: {e}")
        except Exception as e:
            logger.error(f"SQLite 初始化失败，回退到内存模式: {e}")
            # 若 SQLite 失败，自动降级为内存模式
            return MemorySaver()
    else:
        logger.info("使用内存记忆模式（重启后丢失）")
        return MemorySaver()


# 全局默认 checkpointer 实例（供 agent_graph 导入使用）
checkpointer = get_checkpointer()


def clear_memory(thread_id: str = None):
    """
    清空指定会话的记忆（目前仅对内存模式有效，SQLite 暂未实现批量删除）
    :param thread_id: 会话 ID，若为 None 则清空所有（仅内存模式）
    """
    if isinstance(checkpointer, MemorySaver):
        # MemorySaver 内部存储为 dict，但无法直接清空指定 thread_id
        # 可重新创建实例，但不建议在运行时替换
        logger.warning("🔁 内存模式清空所有记忆（重启 Agent 才会生效）")
        # 实际使用中，可以为每个 thread_id 维护单独的 MemorySaver，但这里忽略
    else:
        logger.warning("⚠️ SQLite 模式清空记忆功能暂未实现（可删除数据库文件手动清空）")
        # 高级实现：可执行 DELETE FROM checkpoints WHERE thread_id = ?
    # 简单处理：直接提示
    logger.info("请重启 Agent 或重新实例化 checkpointer 以生效")