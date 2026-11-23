

import numpy as np
from numpy import dot
# 范数
from numpy.linalg import norm
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"), base_url=os.getenv("base_url"))


# 余弦距离
def cos_sim(a, b):
    """  余弦距离   越大越近"""
    return dot(a, b) / (norm(a) * norm(b))


# 欧式距离
def l2(a, b):
    """  距离关系 越小越相识  """
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


def get_embedding(text):
    data = client.embeddings.create(input=text, model='text-embedding-v2')
    return [i.embedding for i in data.data]


query = "我国开展舱外辐射生物学暴露实验"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = get_embedding([query])[0]
# print(query_vec)

doc_vec = get_embedding(documents)
print('余弦距离')
# 余弦距离  完全一样的最大的是1
# print(cos_sim(query_vec, query_vec))
for vec in doc_vec:
    print(cos_sim(query_vec, vec))


print('欧式距离')
# 欧式距离  距离越近  越相似   最像 0
# print(l2(query_vec, query_vec))
for vec in doc_vec:
    print(l2(query_vec, vec))


