import os , json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from azure.search.documents import SearchClient
from src.app.ingestion.pdf_to_markdown import PDFToMarkdownConverter
from azure.core.credentials import AzureKeyCredential
from src.app.ingestion.chunking import read_chunk_upload_docs, parse_company_year
from src.app.rag.kpi_extractor import extract_financial_metrics, Retriever
from src.app.ingestion.azure_storage import upload_blob_data

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

# Input Dir and Output Dir paths
input_dir  = ROOT_DIR / "data" / "raw"
output_dir = ROOT_DIR / "data" / "processed"

def ingest_document(pdf_path : str) -> None:

    #pdf_path = Path(pdf_path)
    
    # Commenting out for now as it already converted to MD and chunks are already uploaded
    PDFToMarkdownConverter(input_dir = pdf_path, output_dir = str(output_dir))
    read_chunk_upload_docs()

    # Initialise Azure AI search client
    search_client = SearchClient(     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
                                      credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
                                      endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
                                 )
   
    retriever =  Retriever(search_client)
   
    company, year = parse_company_year(Path(pdf_path))
    
    metrics =  extract_financial_metrics(retriever, company = company, year = year)
    
    result = upload_blob_data(metrics.model_dump_json(),company,year)
    print(result)

    
if __name__ == "__main__":
    ingest_document(str(input_dir))

# Run this command
# PYTHONPATH=src uv run src/app/ingestion/ingest_documents.py    

# def ingest_directory(input_dir : str) -> None:
#     # Embeddings
#     embeddings = OpenAIEmbeddings(
#                             model= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENNT"),  
#                             dimensions=1024,
#                             api_key=os.getenv("AZURE_OPENAI_API_KEY "),
#                             base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
#                             )
    
   