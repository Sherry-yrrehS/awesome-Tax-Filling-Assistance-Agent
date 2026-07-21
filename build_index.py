from langchain_community.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from variable import VALUE_ADDED_TAX_POLICY_FILE_WORD,VALUE_ADDED_TAX_POLICY_FILE_PDF,COLLECTION_NAME1
from model import Chat_model,Embedding_model
from database import client

#加载文档
all_documents=[]
for file_path in VALUE_ADDED_TAX_POLICY_FILE_WORD:
    loader=UnstructuredWordDocumentLoader(
        file_path=file_path,
        encoding="utf-8"
    )
    documents=loader.load()
    all_documents.extend(documents)

for file_path in VALUE_ADDED_TAX_POLICY_FILE_PDF:
    loader=PyPDFLoader(
        file_path=file_path
    )
    documents=loader.load()
    all_documents.extend(documents)
# print(all_documents)

#切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1100,        # 税法专用长度
    chunk_overlap=220,      # 20%重叠，解决“前款、本条第二项”跨段指代丢失
    separators=[
        "\n第", "\n\n", "\n", "。", "；", " ", ""
    ],
    length_function=len,
    is_separator_regex=False,
    strip_whitespace=True,  # 自动清除多余空行空格
    keep_separator=True  # 保留“第X条”作为chunk的开头
)

chunks=splitter.split_documents(all_documents)

print(f"文档共切分为{len(chunks)}个chunk")

# for i,chunk in enumerate(chunks):
#     print(f"\nchunk{i} : ",chunk.page_content)

#生成向量
text=[
    chunk.page_content for chunk in chunks
]

vectors=Embedding_model.embed_documents(text)

#构建数据
data=[
    {
        "id" : i,
        "vector" : vectors[i],
        "text" : chunks[i].page_content,
        "source" : VALUE_ADDED_TAX_POLICY_FILE_WORD,
        "chunk_id" : i
    }
    for i in range(len(chunks))
]

#插入数据
insert_res = client.upsert(
    collection_name=COLLECTION_NAME1,
    data=data,
)

# print("insert_res")
# print(insert_res)

client.flush(collection_name=COLLECTION_NAME1)

# 打印当前集合中的统计信息
stats = client.get_collection_stats(collection_name=COLLECTION_NAME1)
print(stats)

# 查询当前的collection中有多少条记录

results = client.query(
    collection_name=COLLECTION_NAME1,
    filter="id >= 0",
    output_fields=["id","chunk_id"]
)

print(len(results))