import os 
import uuid 
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_experimental.text_splitter import SemanticChunker


load_dotenv()

UPLOAD_BATCH_SIZE = 500
chunk_size = 300
chunk_overlap = 50

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

# Input Dir and Output Dir paths
input_dir  = ROOT_DIR / "data" / "raw"
output_dir = ROOT_DIR / "data" / "processed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Embeddings
embeddings = OpenAIEmbeddings(
                            model      = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  
                            dimensions = 1024,
                            api_key    = os.getenv("AZURE_OPENAI_API_KEY"),
                            base_url   = os.getenv("AZURE_OPENAI_ENDPOINT")
                            )

# Initialise Azure AI search client
search_client = SearchClient(    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
                                 credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")),
                                 endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
                                 )

# Text Splitter                        
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size    = chunk_size, 
                                                                     chunk_overlap = chunk_overlap,
                                                                     model_name    = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"))

# Use text splitter to split documents
def read_and_chunk_markdown(markdown_file : Path):
    markdown_content = markdown_file.read_text( encoding = "utf-8" )

    return text_splitter.split_text(markdown_content)
    # splitter = SemanticChunker(
    #                             embeddings = embeddings,
    #                             breakpoint_threshold_type = "percentile"
    #                             )
    
    #return splitter.create_documents([markdown_content])


# Function to upload documents in batches
def upload_in_batches(documents, batch_size = UPLOAD_BATCH_SIZE):
    for i in range(0, len(documents), batch_size):
        batch  = documents[i: i+batch_size]
        result = search_client.upload_documents(documents = batch)

        failed = [r for r in result if not r.succeeded]
        if failed:
            logger.warning(
                "%d/%d documents failed in batch %d",
                len(failed), len(batch), i // batch_size,
            )
            for f in failed:
                logger.warning("  key=%s error=%s", f.key, f.error_message)
        else:
            logger.info("Uploaded batch %d (%d docs)", i // batch_size, len(batch))
 

# Get company and year info from file name
def parse_company_year(pdf_file : Path):
    stem = pdf_file.stem
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        year    = parts[0]
        company = parts[-1]
    elif len(parts) >=2 :
        company = parts[0]
        year    = parts[1]
    else:
        company = stem 
        year    = ""

    return company, year


def read_chunk_upload_docs():
    md_files = sorted(Path(output_dir).glob("*.md"))

    if not md_files:
        logger.warning("No .md files found in %s", output_dir)
        return
    
    logger.info("Found %d markdown files in %s", len(md_files), output_dir)
    
    raw_documents = []

    for md_file in md_files:
        company, year = parse_company_year(md_file)
        chunks        = read_and_chunk_markdown(md_file)
        
        for chunk in chunks:
            # vector = embeddings.embed_query(chunk.page_content)
            vector = embeddings.embed_query(chunk)

            # Add meta data to documents
            raw_documents.append(
                {
                    "id"             : str(uuid.uuid4()),
                    "company"        : company,
                    "year"           : year,
                    "source_file"    : md_file.name,
                    "content"        : chunk,
                    "content_vector" : vector
                }
                                
                                )
            
    logger.info("Uploading %d total chunks to Azure AI Search ...", len(raw_documents))        
    upload_in_batches(raw_documents)
    logger.info("Done.")


if __name__ == "__main__":
    read_chunk_upload_docs()




