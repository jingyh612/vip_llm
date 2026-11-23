

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"), base_url=os.getenv("base_url"))



def get_embedding(text):
    data = client.embeddings.create(input=text, model='text-embedding-v2')
    return [i.embedding for i in data.data]



test_query = ['我爱你']

vec = get_embedding(test_query)
print(vec[0])
print(len(vec[0]))


