


# 创建提示模板，配置一个提示模板，将一个示例格式化为字符串
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import langchain_openai
load_dotenv()

# 创建示例
examples = [
    {"input": "2+2", "output": "4", "description": "加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
]
prompt_template = "你是一个数学专家,算式： {input} 值： {output} 使用： {description} "

# 这是一个提示模板，用于设置每个示例的格式
prompt_sample = PromptTemplate.from_template(prompt_template)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt_sample,
    # 告诉大模型要按照这个格式输出description
    suffix="""你是一个数学专家,请计算： {input} 值： {output} """,
    input_variables=["input", "output"],
)

# print(prompt.format(input="3*3", output="9"))

llm = langchain_openai.ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                  model_name='qwen-plus')
result = llm.invoke(prompt.format(input="2*5", output="7"))
print(result.content)  # 使用: 乘法运算




