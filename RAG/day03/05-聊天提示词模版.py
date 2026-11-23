
# 聊天   多了角色应用
# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
# 导入LangChain中的提示模板
# from langchain_core.prompts.chat import ChatPromptTemplate
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
#
#
#
# template = "你是一个数学家，你可以计算任何算式"
# # template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
# human_template = "{text}"
#
#
# chat_prompt = ChatPromptTemplate.from_messages([
#     ('system', template),
#     ('human', human_template)
# ])
#
# print(chat_prompt)




from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()


# 系统模板的构建
system_template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# 用户模版的构建
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


# 组装成最终模版
prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 格式化提示消息生成提示
prompt = prompt_template.format_prompt(input_language="英文", output_language="中文",
                                       text="I love Large Language Model.").to_messages()
# 打印模版
print("prompt:", prompt)






