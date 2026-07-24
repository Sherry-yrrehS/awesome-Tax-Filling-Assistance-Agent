# Milvus客户端，只负责连接和创建（不重建）
from logger_config import setup_logger
from variable import MILVUS_URI,DB_NAME,COLLECTION_NAME1,EMBED_DIM
from pymilvus import MilvusClient,exceptions
import logging
import concurrent.futures
from pymilvus.exceptions import MilvusException

logger = setup_logger("database", log_file="database.log")

#工具函数：带超时的函数执行器
def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """在子线程中执行函数，若超时则抛出 TimeoutError"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"操作超时（{timeout_sec}秒）")

#连接 Milvus（带超时）
def connect_milvus(uri, timeout=10):
    """
    尝试连接 Milvus，若超时或失败则抛出明确异常
    """
    try:
        # 初始化客户端（注意 timeout 单位是秒）
        client = MilvusClient(uri=uri, timeout=timeout)
        # 主动调用一个轻量级接口验证连接是否真正可用
        run_with_timeout(client.list_databases, timeout_sec=timeout)
        logger.info(f"成功连接 Milvus: {uri}")
        return client
    except TimeoutError:
        logger.error(f"连接 Milvus 超时（{timeout}秒），请检查服务是否启动或网络是否可达")
        raise
    except MilvusException as e:  #当 try 代码块抛出 MilvusException 类型错误时，记录异常
        logger.error(f"Milvus 返回异常: {e}")
        raise
    except Exception as e:   #捕获所有常规异常（MilvusException、网络超时、grpc 错误等绝大多数报错都会被接住）
        logger.error(f"连接 Milvus 发生未知错误: {e}", exc_info=True)
        raise


#初始化数据库和集合（带超时）
def init_db_and_collection(client, db_name, collection_name, embed_dim, timeout=30):
    try:
        # 检查/创建数据库
        dbs = run_with_timeout(client.list_databases, timeout_sec=timeout)
        if db_name not in dbs:
            run_with_timeout(client.create_database, timeout_sec=timeout, db_name=db_name)
            logger.info(f"数据库 {db_name} 创建成功")
        run_with_timeout(client.use_database, timeout_sec=timeout, db_name=db_name)
        logger.info(f"已切换到数据库 {db_name}")

        # 检查/创建集合
        has_col = run_with_timeout(client.has_collection, timeout_sec=timeout, collection_name=collection_name)
        if not has_col:
            run_with_timeout(
                client.create_collection,
                timeout_sec=timeout,
                collection_name=collection_name,
                dimension=embed_dim,
                metric_type="COSINE"
            )
            logger.info(f"集合 {collection_name} 创建成功")
        else:
            logger.info(f"集合 {collection_name} 已存在，直接使用")
    except TimeoutError:
        logger.error(f"数据库/集合操作超时（{timeout}秒）")
        raise
    except exceptions.MilvusException as e:
        logger.error(f"Milvus 操作异常: {e}")
        raise
    except Exception as e:
        logger.error(f"初始化数据库/集合失败: {e}", exc_info=True)
        raise


#主流程
try:
    # 连接超时设为 10 秒（可根据网络环境调整）
    client = connect_milvus(MILVUS_URI, timeout=10)

    # 后续操作超时设为 30 秒（因为可能涉及创建集合等稍重操作）
    init_db_and_collection(
        client,
        DB_NAME,
        COLLECTION_NAME1,
        EMBED_DIM,
        timeout=30
    )
except Exception as e:
    logger.critical(f"初始化失败，程序退出: {e}")
    # 可以选择退出或让上层处理
    raise SystemExit(1)