from fastapi import FastAPI, responses, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests, io, fitz, sys
import xml.etree.ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# uvicorn main:app --reload

app = FastAPI()

client = chromadb.Client()
collection = client.create_collection(name="collection1")

model_name = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             dtype = torch.bfloat16,
                                             quantization_config = quantization_config,
                                             device_map="auto")

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
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
    User's Question/Response:
    {query.msg}

    IMPORTANT: ONLY PAY ATTENTION TO THE FOLLOWING SECTIONS IF THEY ARE RELEVANT TO THE USER'S CURRENT QUESTION/RESPONSE

    FOR EXAMPLE, DON'T MAKE THE MISTAKE OF ANSWERING A QUESTION THAT IS ACTUALLY PART OF THE CHAT HISTORY
    OR OVER ANALYZING PAPER DETAILS WHEN THE USER ISN'T EVEN ASKING A QUESTION.
    --------------------------------------------------------------------------------------------------
    1) Details from Paper: 

    {results['documents'][0][0]}

    {results['documents'][0][1]}

    {results['documents'][0][2]}

    --------------------------------------------------------------------------------------------------
    2) Previous Chat History (ai response followed by the user's response):
    
    {query.prevChat}
    """

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print("Created model inputs...", file=sys.stderr)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    print("Created input ids...", file=sys.stderr)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    print("Created output ids...", file=sys.stderr)

    msg = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Decoded output ids.", file=sys.stderr)
    query.msg = msg
    return query