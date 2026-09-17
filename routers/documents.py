import logging
from fastapi import APIRouter, HTTPException
from services.vector_store.factory import get_vector_store
from services.ingestion_service import get_pipeline_v1, get_pipeline_v2, get_pipeline_v3, get_pipeline_v4, ingest
from models.document import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest):
    try:
        vector_store = get_vector_store(request.strategy)
        if request.strategy == "v4":
            pipeline = get_pipeline_v4()
        elif request.strategy == "v3":
            pipeline = get_pipeline_v3()
        elif request.strategy == "v2":
            pipeline = get_pipeline_v2()
        else:
            pipeline = get_pipeline_v1()
        count = ingest(request.directory, vector_store, pipeline, request.strategy)
        return IngestResponse(chunks_ingested=count)
    except ValueError as e:
        logger.error("Ingest failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
