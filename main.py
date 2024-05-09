import io
import json
import os
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from openai import OpenAI
from PyPDF2 import PdfReader
from typing import Optional

import boto3
import pinecone

app = FastAPI()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pinecone.init(api_key = PINECONE_API_KEY, environment = "us-west4-gcp-free")

# DynamoDB configuration
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

dynamodb_client = boto3.client("dynamodb")
dynamodb_resource = boto3.resource("dynamodb")

def insert_question(question):
    clean_question = question.replace("_", " ")
    table = dynamodb_resource.Table('Preguntas')
    response = table.scan()
    unix_timestamp = int(time.time())
    items = response['Items']
    table.put_item(
        Item={
            'id': str(len(items) + 1),
            'Pregunta': clean_question,
            'Timestamp': str(unix_timestamp)
        }
    )

def insert_file(file_name, file_type):
    table = dynamodb_resource.Table('Archivos')
    response = table.scan()
    unix_timestamp = int(time.time())
    items = response['Items']
    table.put_item(
        Item={
            'id': str(len(items) + 1),
            'Nombre': file_name,
            'Tipo': file_type,
            'Timestamp': str(unix_timestamp)
        }
    )

def get_all_data(table_name):
    table = dynamodb_resource.Table(table_name)
    response = table.scan()
    items = response['Items']
    return items

def convert_timestamp(timestamp):
    # Convertir el timestamp a datetime
    dt = datetime.utcfromtimestamp(int(timestamp))
    # Ajustar la hora a la zona horaria de Colombia (GMT-5)
    local_dt = dt - timedelta(hours=5)
    # Formatear la fecha y hora
    formatted_dt = local_dt.strftime('%d-%m-%Y %H:%M:%S')
    return formatted_dt

@app.get("/questions/")
async def questions():
    data = get_all_data('Preguntas')
    decoded_data = []
    for item in data:
        decoded_data.append({
            "id": item["id"],
            #Añadir los signos de pregunta
            "Pregunta": '¿' + item["Pregunta"] + '?',
            "Timestamp": convert_timestamp(item["Timestamp"])
        })
    return decoded_data

@app.get("/files/")
async def files():
    data = get_all_data('Archivos')
    decoded_data = []
    for item in data:
        decoded_data.append({
            "id": item["id"],
            "Nombre": item["Nombre"],
            "Tipo": item["Tipo"],
            "Timestamp": convert_timestamp(item["Timestamp"])
        })
    return decoded_data

@app.get("/search/{q}")
async def search(q: str):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index("langchain-demo", embeddings)
    context = []
    docs = docsearch.similarity_search(q)
    for doc in docs:
        context.append(doc.page_content)

    insert_question(q)

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
                documents.append(Document(page_content=page_text[start:end], metadata={"file": file.filename, "page": page_num}))

                # Actualizar el inicio para el próximo fragmento
                start = end + 1

        vectorstore = Pinecone.from_documents(documents, OpenAIEmbeddings(), index_name="langchain-demo")

        insert_file(file.filename, "PDF")

    return {"filename": file.filename}
