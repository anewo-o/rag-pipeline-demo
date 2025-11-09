from scipy.spatial.distance import cosine
from ollama import Client
import os

client = Client(host='http://localhost:11434')
"""
response = client.chat(model='mistral:7b-instruct', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response.message.content)
"""


def embed(text:str):
    response = client.embed(model='mistral', input=text)
    return response['embeddings'][0]

index = {}
for file in os.listdir(os.path.join(os.getcwd(),"files")):
    index[file] = embed(file)


def search(query: str, index: dict):
    q_emb = embed(query)
    results = [
        (1 - cosine(q_emb, emb), fname)
        for fname, emb in index.items()
    ]
    return sorted(results, reverse=True)

print(search("oiseau", index))

