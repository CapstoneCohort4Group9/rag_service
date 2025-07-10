from fastapi import APIRouter, HTTPException, status
from app.models import QueryRequest, QueryResponse, ErrorResponse
from app.retrieval import retriever
from app.config import settings
import logging

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Submit a query to retrieve relevant documents and generate an AI-powered answer",
    responses={
        200: {"description": "Successful query response"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    }
)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    
    The system will:
    1. Search for relevant documents in the vector database
    2. Use the retrieved context to generate an answer using the Bedrock model
    3. Return the answer along with source documents and confidence score
    """
    try:
        # Validate request
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        if len(request.query) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is too long (maximum 1000 characters)"
            )
        
        # Use default values if not provided
        collection_name = request.collection_name or settings.COLLECTION_NAME
        max_results = min(request.max_results or settings.TOP_K_RESULTS, 20)  # Cap at 20
        similarity_threshold = request.similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process the query
        response = await retriever.query(
            query=request.query,
            collection_name=collection_name,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"Query processed successfully. Found {response.total_sources_found} sources. Confidence: {response.confidence}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/collections",
    summary="List available collections",
    description="Get a list of available vector collections in the database"
)
async def list_collections():
    """
    List all available vector collections in the database.
    """
    try:
        # This is a simple implementation - you might want to query the database
        # for actual collections in production
        return {
            "collections": [settings.COLLECTION_NAME],
            "default_collection": settings.COLLECTION_NAME
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving collections: {str(e)}"
        )