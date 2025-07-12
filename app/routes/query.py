import logging
from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse
from app.retrieval import retrieval_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main RAG query endpoint"""
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process query using retrieval service
        result = await retrieval_service.process_query(request.query, request.max_results)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")