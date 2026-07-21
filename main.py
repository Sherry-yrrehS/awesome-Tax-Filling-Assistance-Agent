from variable import VALUE_ADDED_TAX_POLICY_FILE_WORD,VALUE_ADDED_TAX_POLICY_FILE_PDF,COLLECTION_NAME1
from model import Chat_model,Embedding_model
from database import client
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv(override=True)

#创建Agent
agent=create_agent(
    model=Chat_model,
    tools=[],
    system_prompt=()
)

#检索
# 定义一个具体的函数，实现检索
def retrieve(query : str,limit : int = 3):
    # 将此问题向量化
    query_vector = Embedding_model.embed_query(str(query))
    # print(query_vector)
    # 从向量数据库中检索数据
    results = client.search(
        collection_name=COLLECTION_NAME1,
        data=[query_vector],
        limit=limit,
        output_fields=["text","chunk_id","source"]
    )

    return results[0]

#生成
def generate_answer(query : str):

    # 检索到的数据
    hits = retrieve(str,limit=5)

    # 格式化的操作
    context_blocks = []
    print("=== 检索结果 ===")
    for i, hit in enumerate(hits, 1):
        text = hit["entity"]["text"]
        source = hit["entity"].get("source", "unknown")
        chunk_id = hit["entity"].get("chunk_id", "unknown")
        score = hit["distance"]  # 在 COSINE 模式下，score 越高代表越相似

        print(f"[{i}] chunk_id={chunk_id} score={score:.4f} source={source}")
        print(text)
        print()

        # 拼接成带有编号和元数据的规范上下文块
        context_blocks.append(
            f"[片段{i} | chunk_id={chunk_id} | source={source}]\n{text}"
        )

    # 将多个上下文片段用换行符连成一个大字符串
    context = "\n\n".join(context_blocks)

    # 构造 Prompt
    user_prompt = f"""问题：
{query}

上下文：
{context}
"""
    # 调用agent
    result = agent.invoke({
        "messages" : [{"role": "user","content": user_prompt}],
    })

    final_msg = result["messages"][-1]

    print("====最终回答====")
    final_msg.pretty_print()

q = "个体户是定期定额征收，还需要自己手动申报增值税吗？"
generate_answer(q)