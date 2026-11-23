



# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


# 创建模型实例
model = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model='qwen-plus')


Prompt = PromptTemplate(
    template='你是一个专业的法律问答机器人，你的任务是回答用户的法律问题。用户的问题是：{input}'
)

# 输入提示词
input = Prompt.format(input='帅犯罪吗?')
print(input)


outp = model.invoke(input)
print(outp)









