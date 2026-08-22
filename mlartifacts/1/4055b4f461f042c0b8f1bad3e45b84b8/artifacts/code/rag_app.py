
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Literal
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

ROOT_DIR                  = Path(__file__).parent.parent.parent
PROCESSED_TRANSCRIPTS_DIR = ROOT_DIR / "data" / "processed"
VECTOR_STORE_DIR          = ROOT_DIR / "saved_embeddings"

# Load LLM
llm = ChatOpenAI(
    model    = "gpt-5-mini",  
    api_key  = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
)


# Embeddings
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",  
    dimensions=1024,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

model_name = "gpt-5-mini"
chunk_size = 300
chunk_overlap = 50


# Load documents from the directory
def upsert_documents(chunk_size : int, chunk_overlap : int):
    # Load documents from source
    loader = DirectoryLoader(PROCESSED_TRANSCRIPTS_DIR.as_posix(), 
                         loader_cls=TextLoader, 
                         show_progress=True)

    docs = loader.load()

    # Split documents into tokens 
    # To mirror DeepEval's ContextConstructionnConfig token sizing
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size    = chunk_size, 
                                                                   chunk_overlap = chunk_overlap,
                                                                   model_name    = model_name)
    chunks = chunker.split_documents(docs)

    # Create a Chroma vector store and add the document chunks
    vector_store = Chroma(collection_name="rag_demo",
                        embedding_function = embedder,
                        persist_directory = VECTOR_STORE_DIR.as_posix())

    # Save embbeddings to local
    vector_store.add_documents(chunks)

    return vector_store


# Load Vector Store
def load_knowledge_base():
    # Vector Store
    vector_store = Chroma(collection_name="rag_demo",
                        embedding_function = embedder,
                        persist_directory = VECTOR_STORE_DIR.as_posix())
    return vector_store 


# condition to check if vector store exists and load or create new
if VECTOR_STORE_DIR.exists():
    vector_store = load_knowledge_base()
else:
    vector_store = upsert_documents(chunk_size, chunk_overlap)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3},
                                      search_type = "similarity")


# Define class for the workflow
class RAGState(TypedDict):
    query : str
    response : str
    retrieved_docs : list[Document]
    context : str 
    prompt : ChatPromptTemplate

graph = StateGraph(RAGState)

def retrieve(state: RAGState):
    query = state["query"]
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"context":context,
            "retrieved_docs" : retrieved_docs
            }

def augmentation(state:RAGState):
    query = state["query"]
    context = state["context"]

    prompt = ChatPromptTemplate.from_messages([
        ("system" , ("You are a helpful RAG assistant who responds user query only from given context. \n"
        "If you do not know, say, I don't know. Do not hallucinate or give wrong answers")),
        (("human" , "Context : {context} \n\n  {query}:query"))
                                ])
    return {"prompt":prompt}

def generator(state:RAGState):
    context = state["context"]
    query  = state["query"]
    prompt = state["prompt"]
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context":context, "query":query})

    return {"response":response}
    
    


graph.add_node("retrieve", retrieve)
graph.add_node("augmentation", augmentation)
graph.add_node("generator", generator)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "augmentation")
graph.add_edge("augmentation", "generator")
graph.add_edge("generator", END)

workflow = graph.compile()

answer = workflow.invoke({"query":"How does evolution help in generation of golden"})
print(answer['response'])

