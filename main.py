from typing import Optional
from fastapi import FastAPI
import json
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import pinecone
import os

app = FastAPI()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/search/{q}")
def search(q: str):
    pinecone.init(api_key = "991f3d3f-17f2-47b3-bb46-fa7d3363226a", environment = "us-west4-gcp-free")
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index("langchain-demo", embeddings)
    context = []
    docs = docsearch.similarity_search(q)
    for doc in docs:
        context.append(doc.page_content)

    return context
