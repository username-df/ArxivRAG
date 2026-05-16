from fastapi import FastAPI, responses, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests, io, fitz
import xml.etree.ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core import documents

import chromadb
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal, TypedDict
# uvicorn main:app --reload
# uvicorn main:app --host 0.0.0.0 --port $PORT

load_dotenv()
    
groq_key = os.getenv("GROQ_API_KEY")
groqClient = Groq(api_key=groq_key)

tvly_key = os.getenv("TVLY_API_KEY")
tavily_search = TavilySearch(topic="general", max_results = 1, tavily_api_key=tvly_key)

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

# --------- ARXIV RETRIEVAL AND PARSING --------------------------------------
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
    # use ArXiv API and search using the query
    # return the most recent ArXiv paper
    params = {
    'search_query': f'{query.msg} AND submittedDate:[202401010000 TO 202612312359]',
    'sortBy': 'relevance',
    'sortOrder': 'descending',
    'max_results': 1
    }   

    response = requests.get('https://export.arxiv.org/api/query', params=params)
    print(response.status_code)
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
    docs = [documents.Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    splitted_docs = splitter.split_documents(docs)

    # embed the chunks in a chromaDB collection
    collection.add(
    documents=[chunk.page_content for chunk in splitted_docs],
    ids=[f"id{i}" for i in range(1, len(splitted_docs)+1)]
    )
    return query

# ------------- GRAPH DEFINITION -----------------------------
class GraphState(TypedDict):
    query: str
    context: str
    source: str
    decision: str
    prompt: str
    response: str
    history: str

def get_llm_response(state: GraphState):
    print("------------- GETTING A RESPONSE --------------------------")
    prompt = state["prompt"]

    chat_completion = groqClient.chat.completions.create(
        model="llama-3.3-70b-versatile", 

        messages=[{
            "role": "user",
            "content": prompt,
        }],

        max_tokens=512,        
    )

    state["response"] = chat_completion.choices[0].message.content
    return state

def build_prompt_wcon(state: GraphState):
    print("----------- BUILDING PROMPT W/ CONTEXT -------------------------------")
    print("\nContext:\n")
    print(state["context"])

    prompt = f"""
    ### TASK
    You are a research assistant. Answer the user's question using the context above. 
    - If the context does not contain the answer, ignore the context and answer based on your general knowledge.
    - Be concise.

    ### CONTEXT
    The following segment was retrived from the {state["source"]}. 
    Use it ONLY if it directly answers the user's query.

    [START CONTEXT]
    {state["context"]}
    [END CONTEXT]

    ### CHAT HISTORY
    [START CHAT HISTORY]
    {state["history"]}
    [END CHAT HISTORY]

    User's Question: {state["query"]}
    Response:
    """

    state["prompt"] = prompt
    return state

def build_prompt_ncon(state: GraphState):
    print("---------- BUILDING PROMPT W/ NO CONTEXT ---------------------------------")
    prompt = f"""
    ### CHAT HISTORY
    {state["history"]}

    ### TASK
    You are a research assistant. Answer the user's question using your internal/general knowledge. 
    - Be concise.

    User's Question: {state["query"]}
    Response:
    """

    state["prompt"] = prompt
    return state

def retrieve_context(state: GraphState):
    print("-----RETRIEVING CONTEXT---------------")
    query = state["query"]
    
    results = collection.query(
    query_texts = [query],
    n_results = 3
    )

    context = "\n".join(results["documents"][0])
    state["context"] = context
    print(f"\n CONTEXT: \n{context} \n")
    state["source"] = "ArXiv research paper"
    return state

def tavily_web_search(state: GraphState):
    print("------Performing Web Search-------------")
    query = state["query"]
    result = tavily_search.invoke({"query": query})

    state["context"] = result["results"][0]["content"]
    state["source"] = "Tavily web search"
    return state

def router(state: GraphState) -> Literal["arxiv_paper", "web_search", "internal"]:
    print("---------- ROUTING ------------------")
    decision_prompt = f"""
    You are a router. You must categorize the user's query into exactly one of these three categories: arxiv_paper, web_search, or internal.

    Options:
    - internal: If it's a question that can be answer with your own internal/general knowledge.
    - arxiv_paper: if it's a question that the context from the retrieved ArXiv research paper can answer
    - web_search: If it's about external data (recent news, brand names, etc.) that can not be answered by the paper context or your own general/internal knowledge.
    
    USER'S QUERY: {state["query"]}

    Be concise, your response should simply be one of these three categories: arxiv_paper, web_search, or internal.
    """

    state["prompt"] = decision_prompt
    state = get_llm_response(state)
    router_decision = state["response"].strip().lower()

    print(f"-----------ROUTER DECISION: {router_decision}--------------------")
    state["decision"] = router_decision
    return state

def route_decision(state) -> str:
    return state["decision"]

workflow = StateGraph(GraphState)

workflow.add_node("Generate", get_llm_response)
workflow.add_node("Build_Wcon", build_prompt_wcon)
workflow.add_node("Build_Ncon", build_prompt_ncon)
workflow.add_node("Retrieve_Context", retrieve_context)
workflow.add_node("Web_Search", tavily_web_search)
workflow.add_node("Router", router)

workflow.add_edge(START, "Router")
workflow.add_conditional_edges(
    "Router",
    route_decision,
    {
        "arxiv_paper": "Retrieve_Context",
        "web_search": "Web_Search",
        "internal": "Build_Ncon",
    }
)
workflow.add_edge("Web_Search", "Build_Wcon")
workflow.add_edge("Retrieve_Context", "Build_Wcon")
workflow.add_edge("Build_Wcon", "Generate")
workflow.add_edge("Build_Ncon", "Generate")
workflow.add_edge("Generate", END)
agentic_rag = workflow.compile()

@app.post("/questioning")
async def question(query: Query):    
    input_state = GraphState(query=query.msg, history=query.prevChat)

    for step in agentic_rag.stream(input_state):
        key = list(step.keys())[0]

        if key == 'Generate':
            query.msg = step['Generate']['response']
    return query