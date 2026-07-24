# 加载配置并导出变量
import os
import yaml
from pathlib import Path

# ---------- 重定向模型缓存到 D 盘 ----------
BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / "cache"
CACHE_ROOT.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
os.environ["JIEBA_CACHE"] = str(CACHE_ROOT / "jieba")

# 获取当前文件所在目录（项目根目录）
BASE_DIR = Path(__file__).resolve().parent

# 加载配置文件
with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


#chat模型
CHAT_MODEL_NAME=config["chat_model_name"]

#向量模型
EMBEDDING_MODEL_NAME=config["embedding_model_name"]

#Milvus连接地址
MILVUS_URI=config["milvus_uri"]

#数据库名
DB_NAME=config["db_name"]

#向量数据集名
COLLECTION_NAME1=config["collection_name1"]

#向量嵌入维度
EMBED_DIM=config["embed_dim"]

# 动态生成文件路径列表
DATA_FOLDER = BASE_DIR / config["data_folder"]
VALUE_ADDED_TAX_POLICY_FILE_WORD = [
    str(DATA_FOLDER / fname) for fname in config["file_names"]
]
