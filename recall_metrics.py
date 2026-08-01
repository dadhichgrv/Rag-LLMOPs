import os
from pathlib import Path
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).parent 
EVAL_PATH = ROOT_DIR / "datasets" / "eval_data" / "evaluation_dataset.json"


# llm = ChatOpenAI(
#     model    = "gpt-5-mini",  
#     api_key  = os.getenv("OPENAI_API_KEY"),
#     base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
# )

# metric = ContextualRecallMetric(
#     threshold=0.7,
#     model=llm,
#     include_reason=True
# )

#print(EVAL_PATH.)
# for eval in EVAL_PATH:

#     test_case = LLMTestCase(
#         input=eval.input,
#         actual_output=actual_output,
#         expected_output=expected_output,
#         retrieval_context=retrieval_context
#     )


# recall_metric = ContextualRecallMetric(
#     threshold=0.7,
#     model=llm,
#     include_reason=True
# )

# recall_metric.measure(test_case = test_case)

class FoundryLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-5-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Foundry gpt-5-mini"

test_case = LLMTestCase(
     input             = "Contrast DocLoader, Chunker, Embedder, VecDB+meta for Rack ingest: role/I-O/persistence, note compute/cost.",
     actual_output     = "I will answer only from the provided context.\n\nDocument Loader\n- Role: load transcripts (the source documents) so the chatbot can answer from them.\n- I → O: raw transcripts/files → loaded document objects handed to the next step.\n- Persistence: not specified in the context (I don't know whether loaders persist copies).\n- Compute / cost: not specified (I don't know).\n\nChunker\n- Role: break loaded documents into smaller chunks for downstream embedding and retrieval.\n- I → O: loaded documents → chunks (example in context: 3 transcripts → 542 chunks).\n- Persistence: not specified in the context (I don't know if chunks are persisted outside the next step).\n- Compute / cost: not specified (I don't know).\n\nEmbedder\n- Role: convert chunks into vector embeddings.\n- I → O: chunks → embeddings.\n- Persistence: the context implies embeddings are stored in the vector store, but whether the embedder itself persists outputs is not stated (I don't know).\n- Compute / cost: not specified in the context (I don't know).\n\nVecDB + metadata\n- Role: store embeddings and metadata for retrieval (the vector store used in the example is Chroma).\n- I → O: embeddings + metadata (e.g., collection name) → persistent vector collection used for retrieval (example collection name used: \"rack demo\").\n- Persistence: yes — the vector store is the persistent store for embeddings/metadata (Chroma in the example).\n- Compute / cost: not specified in the context (I don't know).\n\nSummary note from context: ingestion steps are Document Loaders → Chunker → Embedder → VecDB (Chroma example). Many specifics about persistence behavior and compute/cost are not provided in the given context, so I don't know them.",
     expected_output   = "- DocLoader\n  - Role: ingest raw source files (transcript files) from wherever they live (drive, cloud).\n  - I → O: raw files → document strings/records (with source path/ID, timestamps).\n  - Persistence: typically reads from external storage; doesn’t usually persist transformed artifacts itself.\n  - Compute/Cost: low (I/O bound); cheap.\n\n- Chunker (Text Splitter)\n  - Role: break long documents into logical chunks (e.g., 10‑min transcript pieces) for downstream processing.\n  - I → O: document string → chunked texts with chunk metadata (doc id, index, time span).\n  - Persistence: chunks can be persisted (recommended) to avoid re-splitting on repeated runs.\n  - Compute/Cost: minimal CPU; cheap compared with embedding.\n\n- Embedder\n  - Role: convert each chunk to a semantic numeric representation (vectors) that capture meaning.\n  - I → O: chunk text → embedding vectors (usually via external model/API).\n  - Persistence: embeddings should be saved (don’t re‑compute each query).\n  - Compute/Cost: high — uses model compute and/or paid API calls; main cost driver in ingestion.\n\n- VecDB + metadata\n  - Role: persist embeddings and associated metadata (original chunk text, source, timestamps) and serve similarity retrieval.\n  - I → O: vectors + metadata → stored index; queries → retrieved nearest vectors + their metadata/original text.\n  - Persistence: primary persistent store for retrieval; stores vectors and the grounding data needed to produce grounded answers.\n  - Compute/Cost: storage & indexing cost; similarity search cost (much cheaper than repeated embedding calls). By storing embeddings+metadata you avoid repeated expensive embedder calls and keep responses grounded to course transcripts.\n\nKey operational note: keep embeddings and original chunk text (metadata) in the VecDB to ground answers and avoid re‑embedding (major cost savings). DocLoader/Chunker are cheap I/O/CPU steps; Embedder is the costly, compute‑heavy step to minimize and persist.",
     retrieval_context =  [
            "So, the first component in a Rack chatbot\nis\nNow let's talk about the Ingestion phase\nthat is your\nDocument Loaders\nIf you have never heard about Rack\nor if you have not worked with Rack chatbot\ndon't worry about it. I will explain it to you\nwith the first principle approach.\nNow I am understanding what is\ndocument loader?\nWhat we have to do is\nwe have to give transcripts to our chatbot\nand you will answer on the basis of transcripts\nso the work of document loader\nis to load these transcripts",
            "My documents which are my docs\nHere my chunks are created\nIf I want to check here\nHow many chunks are created\nSo\nChunks\nIf I check\nChunks will be my\n542\nSo many chunks\nI have created from my three transcripts\nMeans I have\nShared them\nChunked\nNext what I have to do\nI have to create the vector store\nBecause here\nI have to store\nEmbeddings\nSo I will create the vector store\nVector store will be\nChroma\nHere\nSome things have to be passed\nLike collection name\nLet's keep that\nLet's say rack demo",
            "So, we will not discuss here that what problems does Rack solve?\nIf you want to see that, then you can go to the advanced Rack course.\nI have made a dedicated video that you can see what problems Rack solves there.\nWe just have to understand what are the components of Rack and how they are arranged.\nWe will see that whole thing here.\nSo, Rack is basically our retrieval augmented generation.\nSo, any Rack chatbot, basic, advanced, agentic, graph, Rack, any Rack,\nit has these three steps.\nOkay."
        ]
)

recall_metric = ContextualRecallMetric(
    threshold = 0.7,
    model     = FoundryLLM(),
    include_reason=True
)

recall_metric.measure(test_case = test_case)
print(recall_metric.score)