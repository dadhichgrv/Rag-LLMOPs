from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from src.app.ingestion.azure_storage import get_blob_data
from src.app.vector_store.create_search_index import create_index
from src.routes.chat import router as chatbot_router
from src.routes.dashboard import router as dashboard_router, flatten_metrics
from src.routes.health import router as health_router 
from src.routes.ingestion import router as ingestion_router

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Company Analysis Platform"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        create_index()
    except Exception as e:
        print(f"Warning: Could not create vector index: {e}")
 
    # Preload metrics once at startup so routes don't have to hit Blob
    # Storage on every request.
    try:
        #raw_data = get_blob_data()
        app.state.metrics = get_blob_data()
        print(f"[INFO] preloaded {len(app.state.metrics)} document(s) at startup")
    except Exception as e:
        print(f"Warning: Could not preload metrics: {e}")
        app.state.metrics = []  # safe default so routes don't crash on a missing attribute
 
    yield  # app runs while paused here

app.include_router(ingestion_router,prefix="/api", tags = ["ingestion"])
app.include_router(chatbot_router,prefix="/api", tags = ["chatbot"])
app.include_router(dashboard_router,prefix="/api", tags=["dashboard"])  
app.include_router(health_router, prefix="/api",tags=["health"])  

# app.mount(
#     "/static",
#     StaticFiles(directory="static"),
#     name="static"
# )

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),  
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
def dashboard(request: Request):
    """
    Render dashboard UI.
    """
    metrics = flatten_metrics(get_blob_data())

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "total_companies": len(metrics),
            "total_reports": len(metrics)
        }
    )


@app.get("/metrics")
def metrics():
    """
    Return KPI metrics.
    """
    return JSONResponse(content=get_blob_data() )


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        port=8000
    )