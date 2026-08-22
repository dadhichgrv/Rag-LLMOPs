
import os
from fastapi import APIRouter
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.prompts import ChatPromptTemplate
from src.app.rag.kpi_extractor import extract_financial_metrics, Retriever

router = APIRouter()

# Load LLM
llm = ChatOpenAI(
        model    = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
        api_key  = os.getenv("AZURE_OPENAI_API_KEY"),
        base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                )

class ChatState(BaseModel):
    query    : str 
    company  : str | None = None
    year     : str | None = None

@router.post("/chat")
def chatbot(state: ChatState):
    query   = state.query
    company = state.company
    year    = state.year

    
    # Initialise Azure AI search client
    search_client = SearchClient(     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
                                      credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
                                      endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
                                 )
   
    retriever =  Retriever(search_client)

    docs = retriever.invoke(query, company, year , top_k = 10)
  
    final_context = "\n\n".join(doc.page_content for doc in docs)

    prompt_message = f''' You are an expert financial analyst. Based on given context only , answer user's query . 
                          Do not fabricate any facts or information which is not present in context.
                          In case any information is not available, politely say I don't have enough information.
                      '''

    prompt = ChatPromptTemplate.from_messages([
        ("system" , prompt_message),
        (("human" , "Context : {final_context} \n\n  {query}:query"))
                                ])
    
    rag_chain = prompt | llm 
    response = rag_chain.invoke({"final_context":final_context, "query":query})
    print("Context : ", final_context)
    return response
    



    