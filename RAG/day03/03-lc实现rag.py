

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="text-embedding-v4",
)

# allow_dangerous_deserialization 允许加载 pkl文件
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")

# 创建 LLM 连接（继续使用阿里云 qwen-plus）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 确保环境变量名为 DASHSCOPE_API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 创建llm 文档组合链
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建一个检索器   之前查询用的是向量数据库提供的语法   检索器  封装了查询的方法
retriever = vector_db.as_retriever(search_kwargs={'k': 3})

# 创建检索链
retrieval_chain = create_retrieval_chain(retriever, document_chain)

res = retrieval_chain.invoke({'input': '密云水库水源保护条例什么时候执行'})
print(res)































