#统一日志配置
import logging
import sys
from pathlib import Path


def setup_logger(name, log_file="app.log"):
    """配置并返回一个logger实例"""
    logger = logging.getLogger(name)#logging 采用单例模型：同一个 name 只会生成同一个日志对象，不会重复新建。
    logger.setLevel(logging.INFO)   #设置日志最低接收等级：
                                    # 只会处理 INFO / WARNING / ERROR / CRITICAL；会忽略 DEBUG 调试信息。
                                    # DEBUG < INFO < WARNING < ERROR < CRITICAL

    # 避免重复添加handler
    if logger.handlers:  #logger.handlers是当前日志器挂载的输出通道列表。如果不为空，说明已经配置过控制台/文件输出，直接返回，不再重复创建handler。
        return logger

    # 控制台输出
    console = logging.StreamHandler(sys.stdout)#创建【控制台输出通道】，日志打印到 PyCharm 运行窗口；
    console.setLevel(logging.INFO)#该通道只输出 INFO 及以上；
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')#定义日志模板
    console.setFormatter(console_format)#把格式绑定给控制台处理器。

    # 文件输出
    file_handler = logging.FileHandler(log_file, encoding='utf-8')#创建【文件输出通道】，日志持久化写入磁盘文件；
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)#使用和控制台完全相同的日志格式。

    logger.addHandler(console)
    logger.addHandler(file_handler)#把控制台、文件两个输出通道挂载到 logger；
    return logger                  #返回配置完成的日志对象，供外部文件使用。