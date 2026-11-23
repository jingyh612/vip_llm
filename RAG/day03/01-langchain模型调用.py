
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm =ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen-plus",
)
# 向大模型进行调用
res = llm.invoke('你好')
print(res.content)












