import mlflow
from pathlib import Path

## Run this command "uvx --with "cryptography<49" mlflow server" to run server locally as there is some package issue
## Else generic command is "uvx mlflow server"



# Set tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# # Set experiment name
mlflow.set_experiment("rag_llmops")

# Root DIR
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# start a run
with mlflow.start_run(run_name = "test_run") as run:
    # Log parameters one by one
    mlflow.log_param("chunk_size",300)
    mlflow.log_param("chunk_overlap", 30)
    mlflow.log_param("output_dims", 1024)

    # Log metrics
    mlflow.log_metric("recall",      0.58)
    mlflow.log_metric("faithfulness",0.64)
    mlflow.log_metric("correctness", 0.67)

    # Log rag_app.py file which has main code
    mlflow.log_artifact(local_path = (ROOT_DIR / "src" / "app" / "rag_app.py").as_posix(),
                         artifact_path="code")
    
    # Log golden dataset
    mlflow.log_artifact(local_path=(ROOT_DIR/ 'data' / 'evaluation' / 'goldens' / 'golden_dataset.json').as_posix(),
                        )

# Run this file to start tracking
# uv run src/app/experiment_tracking_basics.py