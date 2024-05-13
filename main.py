from fastapi import FastAPI, File, UploadFile, status, HTTPException
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import io
from PyPDF2 import PdfReader
import os
import boto3
from datetime import datetime, timedelta

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

dynamodb_client = boto3.client("dynamodb")
dynamodb_resource = boto3.resource("dynamodb")

def create_table(table_name):
    table = dynamodb_resource.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'id',
                'KeyType': 'HASH'
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'S'
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)

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

def insert_file(file_name, file_type, file_pages):
    table = dynamodb_resource.Table('Archivos')
    response = table.scan()
    unix_timestamp = int(time.time())
    items = response['Items']
    table.put_item(
        Item={
            'id': str(len(items) + 1),
            'Nombre': file_name,
            'Paginas': file_pages,
            'Tipo': file_type,
            'Timestamp': str(unix_timestamp)
        }
    )

def get_all_filenames():
    table = dynamodb_resource.Table('Archivos')
    response = table.scan()
    items = response['Items']
    filenames = []
    for item in items:
        filenames.append(item['Nombre'])
    return filenames

def delete_file(id):
    table = dynamodb_resource.Table('Archivos')
    table.delete_item(
        Key={
            'id': id
        }
    )

def insert_rating(rating):
    table = dynamodb_resource.Table('Puntuaciones')
    response = table.scan()
    unix_timestamp = int(time.time())
    items = response['Items']
    table.put_item(
        Item={
            'id': str(len(items) + 1),
            'Puntuacion': rating,
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

app = FastAPI()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY") or 'PINECONE_API_KEY'
)

def load_or_create_embeddings_index(index_name, chunks, namespace):
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings...', end='')
        vector_store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), index_name=index_name, namespace=namespace)
        
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        print('Done')
    else:
        print(f'Creating index {index_name} and embeddings ...', end = '')
        pc.create_index(name=index_name, dimension=1536, metric='cosine',  spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            ))
        
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        # Add to vectorDB using LangChain 
        vector_store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), index_name=index_name, namespace=namespace)
        print('Done')   
    return vector_store

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        filenames = get_all_filenames()
        if file.filename in filenames:
            raise HTTPException(status_code=400, detail="File already exists")
        else:
            contents = await file.read()
            pdf_file = io.BytesIO(contents)
            pdf_reader = PdfReader(pdf_file)
            text_splitter = RecursiveCharacterTextSplitter( 
                chunk_size=1000,  # Maximum size of each chunk
                chunk_overlap=100,  # Number of overlapping characters between chunks
            )

            documents = []
            for page_num in range(len(pdf_reader.pages)):
                page_content = pdf_reader.pages[page_num].extract_text()
                documents.append(Document(page_content, metadata={'filename': file.filename, 'page_num': page_num}))

            chunks = text_splitter.split_documents(documents)
            index_name = 'alexa'
            namespace = 'alexa-pdf'

            vector_store = load_or_create_embeddings_index(index_name, chunks, namespace)
            insert_file(file.filename, file.content_type, len(pdf_reader.pages))

            return {"message": "File uploaded successfully"}

@app.post("/rate/")
async def rate(rating: int):
    insert_rating(rating)
    return {"message": "Rating added successfully"}

@app.get("/ratings/")
async def ratings():
    data = get_all_data('Puntuaciones')
    decoded_data = []
    for item in data:
        decoded_data.append({
            "id": item["id"],
            "Puntuacion": item["Puntuacion"],
            "Timestamp": convert_timestamp(item["Timestamp"])
        })
    return decoded_data

@app.get("/search/{q}")
async def search(q: str):
    embeddings = OpenAIEmbeddings()
    
    docsearch = PineconeVectorStore.from_existing_index("alexa", embeddings, namespace="alexa-pdf")
    context = []
    docs = docsearch.similarity_search(q)
    for doc in docs:
        context.append(doc.page_content)

    insert_question(q)

    return context

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
            "Paginas": item["Paginas"],
            "Tipo": item["Tipo"],
            "Timestamp": convert_timestamp(item["Timestamp"])
        })
    return decoded_data

@app.get("/vectors_len/{file_name}")
async def get_vectors(file_name: str):
    index = pc.Index("alexa")
    dummy_vector = [0 for _ in range(1536)]
    ans = index.query(
        vector=dummy_vector,
        filter={
            "filename":{'$eq':file_name}
        },
        top_k=100,
        namespace='alexa-pdf'
    )

    longitud = len(ans['matches'])
    return longitud

def delete_vectors(file_name: str):
    index = pc.Index("alexa")
    dummy_vector = [0 for _ in range(1536)]
    ans = index.query(
        vector=dummy_vector,
        filter={
            "filename":{'$eq':file_name}
        },
        top_k=100,
        namespace='alexa-pdf'
    )

    get_only_ids = lambda x: x['id']

    ids = list(map(get_only_ids, ans['matches']))

    index.delete(ids=ids, namespace='alexa-pdf')

@app.post("/delete_file_vectors/{id}")
async def delete_file_vectors(id: str):
    data = get_all_data('Archivos')
    for item in data:
        if item["id"] == id:
            delete_vectors(item["Nombre"])
            delete_file(id)
            return {"message": "File vectors deleted successfully"}