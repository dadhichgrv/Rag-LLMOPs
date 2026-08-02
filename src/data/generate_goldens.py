import os
from deepeval.synthesizer.synthesizer import Synthesizer 
from deepeval.synthesizer.types import Evolution 
from deepeval.synthesizer.config import FiltrationConfig, EvolutionConfig, ContextConstructionConfig
from pathlib import Path 
from dotenv import load_dotenv

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from deepeval.models import DeepEvalBaseEmbeddingModel



# Load LLM
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


class FoundryEmbedder(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        self.model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def load_model(self):
        return self.model

    def embed_text(self, text: str):
        return self.load_model().embed_query(text)

    def embed_texts(self, texts: list):
        return self.load_model().embed_documents(texts)

    async def a_embed_text(self, text: str):
        return await self.load_model().aembed_query(text)

    async def a_embed_texts(self, texts: list):
        return await self.load_model().aembed_documents(texts)

    def get_model_name(self):
        return "Foundry text-embedding-3-small"



load_dotenv()

# define paths
# Path(__file__) will give path of current file but i want the path of parent folder , then add .parent
ROOT_PATH   = Path(__file__).parent.parent.parent
DOCS_PATH = ROOT_PATH / "data" / "processed"
GOLDENS_DIR = ROOT_PATH / "data" / "evaluation" / "goldens"

# Create directory to store goldens
GOLDENS_DIR.mkdir(exist_ok=True, parents=True)

def get_docs_path(directory_path: Path | str):
    # Directory path can be path object or string. So we are convevrting it to Path object anyways
    directory_path = Path(directory_path)
    # Check if dir path exists or is directory
    if directory_path.exists() and directory_path.is_dir():
        # search for all .txt files in that path
        paths = directory_path.glob('*.txt')
        # return all paths as string 
    return [path.as_posix() for path in paths]
    

# Set filtrationn config
filtration_config = FiltrationConfig(
                                     critic_model                       = FoundryLLM(),
                                     max_quality_retries                = 2,
                                     synthetic_input_quality_threshold  = 0.5
                                     )



# Evolution config
evolution_config = EvolutionConfig(
                    evolutions = {
                         Evolution.MULTICONTEXT : 1/4,
                         Evolution.CONCRETIZING : 1/4,
                         Evolution.CONSTRAINED  : 1/4,
                         Evolution.COMPARATIVE  : 1/4 
                                 },
                        num_evolutions=3
                            )


context_config = ContextConstructionConfig(
    embedder=FoundryEmbedder(),          # moved here, not on Synthesizer
    critic_model=FoundryLLM(),
    max_contexts_per_document=1,
    chunk_overlap = 50,
    chunk_size    = 300,
    max_context_length=2,
    context_quality_threshold=0.3,
    max_retries=2,
)


# Synthesizer
synthesizer = Synthesizer(
                          model             = FoundryLLM(),
                          filtration_config = filtration_config,
                          evolution_config  = evolution_config,
                          max_concurrent    = 2,   # 2 goldens to be generated in parallel. If this number is big, you may get rate limit error
                          )


# Generate goldens
goldens = synthesizer.generate_goldens_from_docs(
          document_paths              = get_docs_path(DOCS_PATH),
          max_goldens_per_context     = 1,
          include_expected_output     = True,         # This will generate the expected answer also and is default to true
          context_construction_config = context_config
          )          
                                                



synthesizer.save_as(
    file_type = 'json',
    directory = GOLDENS_DIR.as_posix(),
    file_name = "golden_dataset"
                    )

