
import os 
from pathlib import Path
from fastapi import APIRouter, File, UploadFile
from langchain_openai import OpenAIEmbeddings
from azure.search.documents import SearchClient
from src.app.ingestion.ingest_documents import ingest_document

router = APIRouter()

ROOT_PATH = Path(__file__).parent.parent.parent.resolve()

# embeddings = OpenAIEmbeddings(
#                             model= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENNT"),  
#                             dimensions=1024,
#                             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#                             base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
#                             )

# # Initialise Azure AI search client
# search_client = SearchClient(    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
#                                  credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
#                                  endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
#                                  )

@router.post("/upload")
def upload_document(file : UploadFile = File(...)):
    upload_dir = Path(ROOT_PATH / "data" / "raw")
    upload_dir.mkdir(parents = True, exist_ok = True)
    
    # This is the path for file to be uplaoded
    file_path = upload_dir / file.filename

    ingest_document(pdf_path = str(file_path))
    
    return {"message" : "Document Uploaded Successfully",
            "file_name" : file.filename}

    
