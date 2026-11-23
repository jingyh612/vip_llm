
# 可以直接获取静态数据
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
# langchain_community.embeddings  存放了所有第三方的向量嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings
# langchain_community.vectorstores  存的第三放向量数据库
from langchain_community.vectorstores import FAISS





loader = WebBaseLoader(
    'https://www.gov.cn/zhengce/content/202510/content_7043916.htm',
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id='UCAP-CONTENT'))
)
docs = loader.load()
# 返回分割器对象
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
documents = text_splitter.split_documents(docs)
print(len(documents))


embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="text-embedding-v4",
)

# 把文档转换成向量然后存在向量数据库中 batch size  批次  同时处理向量的大小

for i in range(0, len(documents), 10):
    batch_docs = documents[i:i+10]
    print(f'第{i}批次  文档数量:', len(batch_docs))
    vector = FAISS.from_documents(batch_docs, embeddings)
    # 保持到本地
    vector.save_local("faiss_index")










