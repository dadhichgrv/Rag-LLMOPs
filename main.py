
deepeval set-azure-openai \
    --base-url = "https://ai-founndry-demo.services.ai.azure.com/openai/v1/" \ 
    --model    = "gpt-5-mini"
    --deployment-name=<deployment_name> \  # e.g. Test Deployment
    --api-version=<api_version> \ # e.g. 2025-01-01-preview
    --model-version=<model_version> # e.g. 2024-11-20