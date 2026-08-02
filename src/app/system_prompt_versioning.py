from langfuse import get_client 
from dotenv import load_dotenv

load_dotenv()

chunk_size = 300
chunk_overlap = 30
output_dimensions = 1024
k = 3


system_prompt = """
You are a helpful RAG assistant who responds user query only from given context. 
If you do not know, say, I don't know. Do not hallucinate or give wrong answers
"""

# Initialize Langfuse client
langfuse = get_client()

# Create a text prompt
created_prompt = langfuse.create_prompt(
                             name   = "rag_app_system_prompt",
                             type   = "text",
                             prompt = system_prompt,
                             labels = ["baseline"],
                             config = {
                                        "chunk_size"    : chunk_size,
                                        "chunk_overlap" : chunk_overlap,
                                        "output_dims"   : output_dimensions,
                                        "k" : k
                                          }  
                                       )


print(created_prompt.prompt)
print(created_prompt.version)
print(created_prompt.labels)