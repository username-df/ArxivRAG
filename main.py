import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, responses, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests, io, fitz, sys
import xml.etree.ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import chromadb
import os
from dotenv import load_dotenv
from groq import Groq
# uvicorn main:app --reload

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
groqClient = Groq(api_key=api_key)

chromaClient = chromadb.Client()
collection = chromaClient.create_collection(name="collection1")
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", staticfiles.StaticFiles(directory="static"), name="static")
@app.get("/")
async def read_index():
    return responses.FileResponse('index.html')

class Query(BaseModel):
    msg: str
    prevChat: str

def find_reference_page(doc):
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip().lower()

            if text == "references":
                return page_num
    return len(doc)

@app.post("/retrieval")
async def retrieve(query: Query):
    # use Arxiv API and search using the query
    params = {
    'search_query': f'{query.msg} AND submittedDate:[202301010000 TO 202612302359]',
    'sortBy': 'relevance',
    'sortOrder': 'descending',
    'max_results': 1
    }   

    response = requests.get('https://export.arxiv.org/api/query', params=params)
    print(response.status_code, file=sys.stderr)
    result = ET.fromstring(response.text).findall('.//{*}link[@title="pdf"]')
    query.msg = (ET.fromstring(response.text).findall('.//{*}title'))[1].text

    # extract text from pdf link, excluding references
    pdf = io.BytesIO((requests.get(result[0].attrib['href']).content))
    with fitz.open(stream=pdf) as doc:
        ref_page = find_reference_page(doc)

        text = ""
        for i in range(0, ref_page):
            text += doc[i].get_text()

    # use recursive chunking on the pdf text
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    splitted_docs = splitter.split_documents(docs)

    collection.add(
    documents=[chunk.page_content for chunk in splitted_docs],
    ids=[f"id{i}" for i in range(1, len(splitted_docs)+1)]
    )
    return query


@app.post("/questioning")
async def question(query: Query):    
    results = collection.query(
    query_texts = [query.msg],
    n_results = 3
    )

    prompt = f"""
    ### CONTEXT DOCUMENTS
    The following segments are retrieved from a research paper. 
    Use them ONLY if they directly answer the user's query.

    1) {results['documents'][0][0]}
    2) {results['documents'][0][1]}
    3) {results['documents'][0][2]}

    ### CHAT HISTORY
    {query.prevChat}

    ### TASK
    You are a research assistant. Answer the user's question using the context above. 
    - If the context does not contain the answer, ignore the context and answer based on your general knowledge.
    - Be concise.

    User's Question: {query.msg}
    Response:
    """

    chat_completion = groqClient.chat.completions.create(
        model="llama-3.3-70b-versatile", 

        messages=[{
            "role": "user",
            "content": prompt,
        }],

        max_tokens=512,        
    )

    query.msg = chat_completion.choices[0].message.content
    return query