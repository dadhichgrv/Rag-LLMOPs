from deepeval.test_case import LLMTestCase 
from deepeval.dataset import EvaluationDataset, Golden 
from pathlib import Path
from dotenv import load_dotenv
from src.app.rag_app import workflow
from time import sleep

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent
GOLDENS_PATH = ROOT_DIR / "data" / "evaluation" / "goldens" / "golden_dataset.json"

# create dir for storing ealuation dataset
EVAL_DATA_DIR = ROOT_DIR / "data" / "evaluation" / "eval_dataset"
EVAL_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Read Goldens from Goldenn Path
golden_dataset = EvaluationDataset()
# Add goldens to eval dataset
golden_dataset.add_goldens_from_json_file(
                    file_path=GOLDENS_PATH
                                  )

# Datset to hold produced test cases
eval_dataset = EvaluationDataset()
# Create test cases for each golden 
for golden in golden_dataset.goldens:
    final_state = workflow.invoke({"query" : golden.input})
    sleep(2)
    test_case = LLMTestCase(
                        input             = golden.input,
                        actual_output     = final_state.get("response"),
                        expected_output   = golden.expected_output,
                        retrieval_context = [doc.page_content for doc in final_state.get("retrieved_docs")]
                            )
    
    # Here we are appending this LLM Test Case to our golden data so you will see these test cases after 20 goldens in ealuation_dataset.json
    eval_dataset.add_test_case(test_case = test_case)


eval_dataset.save_as(
                file_type = 'json',
                directory= EVAL_DATA_DIR,
                file_name="evaluation_dataset",
                include_test_cases=True
                    )
    


    





