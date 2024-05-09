import io
import json
import os
import time

from fastapi import FastAPI, File, UploadFile
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from openai import OpenAI
from PyPDF2 import PdfReader
from typing import Optional

import pinecone
import boto3

app = FastAPI()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pinecone.init(api_key = PINECONE_API_KEY, environment = "gcp-starter")

# DynamoDB configuration
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
REGION_NAME = os.environ["REGION_NAME"]

dynamodb_client = boto3.client("dynamodb")
dynamodb_resource = boto3.resource("dynamodb")


def insert_data(question):
    table = dynamodb_resource.Table('Preguntas')
    response = table.scan()
    unix_timestamp = int(time.time())
    items = response['Items']
    table.put_item(
        Item={
            'id': str(len(items) + 1),
            'Pregunta': question,
            'Timestamp': str(unix_timestamp)
        }
    )

def get_all_data():
    table = dynamodb_resource.Table('Preguntas')
    response = table.scan()
    items = response['Items']
    return items

@app.get("/search/{q}")
async def search(q: str):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index("alexa", embeddings)
    context = []
    docs = docsearch.similarity_search(q)
    for doc in docs:
        context.append(doc.page_content)

    pregunta = q

    insert_data(pregunta)

    return context

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        pdf_reader = PdfReader(pdf_file)
        chunk_size = 1000  # Define el tamaño de los fragmentos

        documents = []  # Lista para almacenar los objetos Document

        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()

            # Dividir el texto de la página en fragmentos
            start = 0
            while start < len(page_text):
                tmp_end = min(start + chunk_size, len(page_text))
                dot_pos = page_text[:tmp_end].rfind('.')
                space_pos = page_text[:tmp_end].rfind(' ')
                change = tmp_end != len(page_text)
                end = max(dot_pos, space_pos) if change else tmp_end

                # Crear un objeto Document con el fragmento de texto y añadirlo a la lista
                documents.append(Document(page_content=page_text[start:end]))

                # Actualizar el inicio para el próximo fragmento
                start = end + 1

        vectorstore = Pinecone.from_documents(documents, OpenAIEmbeddings(), index_name="alexa")

    return {"filename": file.filename, "text": [doc.page_content for doc in documents]}
