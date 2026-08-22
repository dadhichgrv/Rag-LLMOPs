import os
from types import SimpleNamespace
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load LLM
llm = ChatOpenAI(
    model    = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
    api_key  = os.getenv("AZURE_OPENAI_API_KEY"),
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                )


# Load Azure Search Client
search_client = SearchClient(    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
                                 credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
                                 endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
                                 )

# Define schema 
class kpi_schema(BaseModel):
    revenue : str | float | None = Field(default=None, description="Total or net revenue from income statement")
    profit  : str | float | None = Field(default=None, description="Net income or net profit")
    operating_income : str | float | None = Field(default=None)
    cash_flow : str | float | None = Field(default=None, description="Cash flow from operations")
    total_asset : str | float | None = Field(default=None, description="Total assets from balance sheet")
    total_liabilities : str | float | None = Field(default=None, description="Total liabilities from balance sheet")
    risk_factors : str | None = Field(default=None)
    growth_drivers : str | None = Field(default=None)


# Pass schema to LLM
llm_with_structured_output = llm.with_structured_output(kpi_schema)


# Define Retriever class
class Retriever:
    """Hybrid (vector + keyword) retriever over the Azure AI Search index."""
    def __init__(self, client = search_client):
        self.client = client

    def invoke(self, query : str, company : str | None = None, year : int | None = None , top_k : int = 10):

        filter_expression = None 

        if company and year:
            filter_expression = (f""" company eq '{company}' and year eq '{year}' """)
            
        if filter_expression:
            results = self.client.search(
                                    search_text = query,
                                    top = top_k,
                                    filter = filter_expression
                                    )
        else:
            results = self.client.search(
                                    search_text = query,
                                    top = top_k
                                        )
            
        documents = []

        for result in results:
            content = result.get("content","")
            chunk_id = result.get("id") or result.get("metadata_storage_name") or str(hash(content))
            documents.append(SimpleNamespace(page_content=content, id=chunk_id))
            #documents.append(SimpleNamespace(page_content = content))

        return documents


# Define function to retrieve context 
def retrieve_context(retriever: Retriever, company : str, year : int) -> str:
    """Runs targeted searches for specific statements to maximize search accuracy."""

    # Define multiple queries onne for each metric
    target_queries = [
        "total revenue net revenue net income profit operating income statement",
        "cash flow from operations statement of cash flows",
        "total assets total liabilities balance sheet",
        "risk factors item 1A uncertainties",
        "growth drivers strategy business opportunities outlook"
    ]
    
    unique_contents = {}
    
    # Run targeted mini-retrievals instead of one large query
    for sub_query in target_queries:
        # Request top n chunks for each targeted section to prevent token flooding
        docs = retriever.invoke(sub_query, company, year, top_k = 10)
        for doc in docs:
            unique_contents[doc.id] = doc.page_content
            
    return "\n\n".join(unique_contents.values())



def extract_financial_metrics(retriever: Retriever, company : str, year : int):
    # Get retrieved context
    context = retrieve_context(retriever, company, year)
   
    prompt = f""" 
    You are a professional financial analyst.
    Review the context below carefully for company '{company}' and year {year}.
    
    Context:
    {context}
    
    Extract all requested metrics using ONLY the provided context. 
    - Do not invent, speculate, or extrapolate values.
    - If a specific metric value is completely absent from the context, leave it blank or null.
    - Risk Factors and Growth Drivers must be highly concise bullet points or short summaries.
    """

    metrics = llm_with_structured_output.invoke(prompt)

    return metrics


def main():
    retriever = Retriever(search_client)
    metrics =  extract_financial_metrics(retriever, company = "MSFT", year = 2026)

    print("\n--- Extracted Financial Metrics ---")
    for key, value in metrics.model_dump().items():
        print(f"{key.replace('_', ' ').title()}: {value}")


if __name__=="__main__":
    main()







