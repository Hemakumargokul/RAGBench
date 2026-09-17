import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import chat, documents
from services.llm.factory import get_llm
from services.vector_store.factory import get_vector_store
from services.ingestion_service import get_pipeline_v1, get_pipeline_v2, get_pipeline_v3, get_pipeline_v4
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    force=True
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger = logging.getLogger(__name__)
    logger.info("Initializing dependencies...")
    get_llm()
    get_vector_store("v1")
    get_vector_store("v2")
    get_vector_store("v3")
    get_vector_store("v4")
    get_pipeline_v1()
    get_pipeline_v2()
    get_pipeline_v3()
    get_pipeline_v4()
    logger.info("All dependencies ready.")
    yield

app = FastAPI(title="RAGBench API", version="1.0.0", lifespan=lifespan)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
