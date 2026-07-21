from variable import MILVUS_URI,DB_NAME,COLLECTION_NAME1,EMBED_DIM
from pymilvus import MilvusClient

#初始化Milvus
client=MilvusClient(MILVUS_URI)
db_name=DB_NAME

#查询已有数据库，若不存在则创建
existed_databases=client.list_databases()
if db_name not in existed_databases:
    client.create_database(db_name)

#切换到指定数据库
client.use_database(db_name)

#创建collection1增值税收
if client.has_collection(collection_name=COLLECTION_NAME1):
    client.drop_collection(collection_name=COLLECTION_NAME1)

client.create_collection(
    collection_name=COLLECTION_NAME1,
    dimension=EMBED_DIM,
    metric_type="COSINE"
)
# print(client.list_databases())
# print(client.list_collections())



