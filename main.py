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


# {
#     "places": [
#         {
#             "name": "Modern Art Loft",
#             "rating": 4.5,
#             "location": {
#                 "country": "United States",
#                 "city": "New York"
#             },
#             "image_url": "https://cdn.tiqets.com/wordpress/blog/wp-content/uploads/2017/08/03134557/24-hours-in-new-york-1-1024x570.jpg",
#             "date": "2023-01-15",
#             "price": 120.99
#         },
#         {
#             "name": "Adventure National Park Lodge",
#             "rating": 4.2,
#             "location": {
#                 "country": "Canada",
#                 "city": "Vancouver"
#             },
#             "image_url": "https://www.vmcdn.ca/f/files/via/images/city-images/downtown-vancouver-sunny-day-shore.jpg;w=960;h=586;mode=crop",
#             "date": "2023-02-20",
#             "price": 95.75
#         },
#         {
#             "name": "Seafront Restaurant Retreat",
#             "rating": 4.8,
#             "location": {
#                 "country": "United Kingdom",
#                 "city": "London"
#             },
#             "image_url": "https://plus.unsplash.com/premium_photo-1661964203376-9aeba8cdbd23?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8bG9uZG9uJTJDJTIwdW5pdGVkJTIwa2luZ2RvbXxlbnwwfHwwfHx8MA%3D%3D",
#             "date": "2023-03-10",
#             "price": 225.5
#         },
#         {
#             "name": "Mountain Trail Cabin",
#             "rating": 4,
#             "location": {
#                 "country": "Australia",
#                 "city": "Sydney"
#             },
#             "image_url": "https://images.squarespace-cdn.com/content/v1/55ee34aae4b0bf70212ada4c/1577545161018-1F9Z9ZZQG9JO2O4WCWQX/keith-zhu-qaNcz43MeY8-unsplash+%281%29.jpg?format=1500w",
#             "date": "2023-04-05",
#             "price": 75.99
#         },
#         {
#             "name": "Downtown Theater Studio",
#             "rating": 4.7,
#             "location": {
#                 "country": "United States",
#                 "city": "Los Angeles"
#             },
#             "image_url": "https://cdn.britannica.com/97/100097-050-7E411D6F/Harbor-Freeway-Los-Angeles.jpg",
#             "date": "2023-05-12",
#             "price": 150.25
#         }
#     ]
# }

@app.get("/places")
def places():
    with open("places.json") as f:
        places = json.load(f)
    return places