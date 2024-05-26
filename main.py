from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fastapi.responses import RedirectResponse
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from botocore.exceptions import ClientError
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
s3 = boto3.client('s3')
bucket_name = os.environ["BUCKET_NAME"]

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
    if response['Items']:
        max_id = max(int(item['id']) for item in response['Items'])
    else:
        max_id = 0
    table.put_item(
        Item={
            'id': str(max_id + 1),
            'Pregunta': clean_question,
            'Timestamp': str(unix_timestamp)
        }
    )

def insert_file(file_name, file_type, file_pages):
    table = dynamodb_resource.Table('Archivos')
    response = table.scan()
    unix_timestamp = int(time.time())
    if response['Items']:
        max_id = max(int(item['id']) for item in response['Items'])
    else:
        max_id = 0
    table.put_item(
        Item={
            'id': str(max_id + 1),
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
    if response['Items']:
        max_id = max(int(item['id']) for item in response['Items'])
    else:
        max_id = 0
    table.put_item(
        Item={
            'id': str(max_id + 1),
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
    formatted_dt = local_dt.strftime('%d-%m-%Y %H:%M')
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

def create_bucket(bucket_name):
    bucket_name = bucket_name.lower()
    if bucket_name not in [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]:
        try:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={
                'LocationConstraint': AWS_DEFAULT_REGION
            })
            print(f"Bucket {bucket_name} created")
        except Exception as e:
            print(f"Error creating bucket: {e}")
    else:
        print(f"Bucket {bucket_name} already exists in the account")

def list_files_in_bucket(bucket_name):
    if bucket_name in [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]:
        try:
            response = s3.list_objects_v2(Bucket=bucket_name)
            if 'Contents' in response:
                for obj in response['Contents']:
                    print(obj['Key'])
            else:
                print(f"Bucket {bucket_name} is empty")
        except Exception as e:
            print(f"Error listing files: {e}")
    else:
        print(f"Bucket {bucket_name} does not exist in the account")
    
def insert_binary_file_into_bucket(bucket_name, binary_file, file_name):
    if bucket_name in [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]:
        try:
            s3.put_object(Bucket=bucket_name, Key=file_name, Body=binary_file)
            print(f"File {file_name} uploaded to bucket {bucket_name}")
        except Exception as e:
            print(f"Error uploading file: {e}")
    else:
        print(f"Bucket {bucket_name} does not exist in the account")

def delete_file_from_bucket(bucket_name, file_name):
    if bucket_name in [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]:
        try:
            s3.delete_object(Bucket=bucket_name, Key=file_name)
            print(f"File {file_name} deleted from bucket {bucket_name}")
        except Exception as e:
            print(f"Error deleting file: {e}")
    else:
        print(f"Bucket {bucket_name} does not exist in the account")

def presigned_url(bucket_name, object_name, expiration=3600):
    try:
        response = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': f'information_files/{object_name}'}, ExpiresIn=expiration)
        return response
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        if 'Archivos' not in dynamodb_client.list_tables()['TableNames']:
            create_table('Archivos')
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
            if bucket_name not in [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]:
                create_bucket(bucket_name)
            
            insert_binary_file_into_bucket(bucket_name, contents, f'information_files/{file.filename}')

            return {"message": "File uploaded successfully"}

@app.get("/downloadfile/{file_name}")
async def downloadfile(file_name: str):
    if 'Archivos' not in dynamodb_client.list_tables()['TableNames']:
        return {"message": "No files table found, upload a file first"}
    data = get_all_data('Archivos')
    for item in data:
        if item["Nombre"] == file_name:
            return presigned_url(bucket_name, item["Nombre"])
    return {"message": "File not found"}

@app.post("/rate/")
async def rate(rating: int):
    #Si no existe la tabla, crearla
    if 'Puntuaciones' not in dynamodb_client.list_tables()['TableNames']:
        create_table('Puntuaciones')
    insert_rating(rating)
    return {"message": "Rating inserted successfully"}

@app.get("/ratings/")
async def ratings():
    if 'Puntuaciones' not in dynamodb_client.list_tables()['TableNames']:
        return {"message": "No ratings table found, post a rating first"}
    data = get_all_data('Puntuaciones')
    decoded_data = []
    for item in data:
        decoded_data.append({
            "id": item["id"],
            "Puntuacion": item["Puntuacion"],
            "Timestamp": convert_timestamp(item["Timestamp"])
        })
    sorted_data = sorted(decoded_data, key=lambda x: x["Timestamp"], reverse=True)
    return sorted_data

@app.get("/search/{q}")
async def search(q: str):
    embeddings = OpenAIEmbeddings()
    
    docsearch = PineconeVectorStore.from_existing_index("alexa", embeddings, namespace="alexa-pdf")
    context = []
    docs = docsearch.similarity_search(q)
    for doc in docs:
        context.append(doc.page_content)

    if 'Preguntas' not in dynamodb_client.list_tables()['TableNames']:
        create_table('Preguntas')
    insert_question(q)

    return context

@app.get("/questions/")
async def questions():
    if 'Preguntas' not in dynamodb_client.list_tables()['TableNames']:
        return {"message": "No questions table found, post a question first"}
    data = get_all_data('Preguntas')
    decoded_data = []
    for item in data:
        decoded_data.append({
            "id": item["id"],
            #Añadir los signos de pregunta
            "Pregunta": '¿' + item["Pregunta"] + '?',
            "Timestamp": convert_timestamp(item["Timestamp"])
        })
    sorted_data = sorted(decoded_data, key=lambda x: x["Timestamp"], reverse=True)
    return sorted_data

@app.get("/files/")
async def files():
    if 'Archivos' not in dynamodb_client.list_tables()['TableNames']:
        return {"message": "No files table found, upload a file first"}
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
    sorted_data = sorted(decoded_data, key=lambda x: x["Timestamp"], reverse=True)
    return sorted_data

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
            delete_file_from_bucket(bucket_name, item["Nombre"])
            return {"message": "File vectors deleted successfully"}