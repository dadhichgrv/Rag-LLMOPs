import os
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics import (GEval, 
                              AnswerRelevancyMetric,
                              FaithfulnessMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric)

from deepeval.test_case.llm_test_case import SingleTurnParams
from langchain_openai import ChatOpenAI
from deepeval.evaluate import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, CacheConfig
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
        



# Define Metrics
recall               = ContextualRecallMetric(model     = FoundryLLM())
precision            = ContextualPrecisionMetric(model     = FoundryLLM())
contextual_relevancy = ContextualRelevancyMetric(model     = FoundryLLM())
answer_relevancy     = AnswerRelevancyMetric(model     = FoundryLLM())
faithfulness         = FaithfulnessMetric(model     = FoundryLLM())

# Define custom metric
answer_correctness = GEval(
    model = FoundryLLM(),
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    rubric=[
        Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
        Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10,10), expected_outcome="100% correct."),
           ]
                           )


# define evaluation dataset path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
evaluation_dataset_path = ROOT_DIR / 'data' / 'evaluation' / 'eval_dataset' / 'evaluation_dataset.json' 

if evaluation_dataset_path.exists():
    # load dataset
    evaluation_dataset = EvaluationDataset()

    evaluation_dataset.add_test_cases_from_json_file(
        file_path = evaluation_dataset_path,
        input_key_name= "input",
        actual_output_key_name = "actual_output",
        expected_output_key_name= "expected_output",
        retrieval_context_key_name= "retrieval_context"
    )
    

    # Store test cases in list
    test_cases = evaluation_dataset.test_cases

    # Evaluate dataset
    evaluate(test_cases=test_cases,
             metrics = [recall,
                        precision,
                        contextual_relevancy,
                        answer_relevancy,
                        faithfulness,
                        answer_correctness],
            async_config = AsyncConfig(throttle_value = 3, max_concurrent = 5),

            # Evaluation Results are LLM Readable 
            # Evaluatio Report is human readable so use that
            display_config = DisplayConfig(results_folder = (ROOT_DIR / "reports" / "evaluation_results").as_posix(),
                                           file_type="md",
                                           file_output_dir=(ROOT_DIR / "reports" / "evaluation_report").as_posix()
                                          ) ,
            cache_config = CacheConfig(write_cache = True, use_cache = True)
            )
    

# Cmd + Shift + V to view in .MD format in VSCode