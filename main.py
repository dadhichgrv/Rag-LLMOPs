from fastapi import FastAPI

from src.routes.chat import router as chatbot_router
from src.routes.dashboard import router as dashboard_router
from src.routes.health import router as health_router 
from src.routes.ingestion import router as ingestion_router

app = FastAPI(title = "Financial API")

app.include_router(health_router, tags = ["health"])
app.include_router(dashboard_router, tags = ["dashboard"])
app.include_router(chatbot_router, tags = ["chatbot"])
app.include_router(ingestion_router, tags = ["ingestion"])

if __name__=="__main__":
    import uvicorn

    uvicorn.run("main:app", port = 8080, reload = True)
