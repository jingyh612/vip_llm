

# 下载模型到本地
# from modelscope import snapshot_download
# # pip install modelscope
# model_dir = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir=r"D:\LLM\Local_model")
# 加载模型
from sentence_transformers import SentenceTransformer

# 加载本地模型   嵌入模型  把数据转换成向量
model = SentenceTransformer(r'D:\LLM\Local_model\BAAI\bge-large-zh-v1___5')


data = ['你好']
# 默认返回的nump数组
embedding = model.encode(data)
print(embedding.shape)




