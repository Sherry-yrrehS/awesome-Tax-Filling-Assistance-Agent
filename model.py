import os
import dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from variable import CHAT_MODEL_NAME,EMBEDDING_MODEL_NAME

dotenv.load_dotenv()

Chat_model=ChatOpenAI(
    model=CHAT_MODEL_NAME,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)
# print(llm.invoke("1+1=?"))

Embedding_model=OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL")
)