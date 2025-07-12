from fastapi import APIRouter

from app.models import HealthResponse
from app.bedrock import test_bedrock_connection
from app.database import test_database_connection
from app.embeddings import is_embeddings_ready

router = APIRouter()

@router.get("/health-deep", response_model=HealthResponse)
async def health_deep():
    """Comprehensive health check"""
    
    # Test Bedrock
    bedrock_status = "healthy" if test_bedrock_connection() else "unhealthy"
    
    # Test Database
    database_status = "healthy" if test_database_connection() else "unhealthy"
    
    # Test Embeddings
    embeddings_status = "healthy" if is_embeddings_ready() else "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all([
        bedrock_status == "healthy",
        database_status == "healthy", 
        embeddings_status == "healthy"
    ]) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        database_status=database_status,
        bedrock_status=bedrock_status,
        embeddings_status=embeddings_status
    )

@router.get("/health")
async def health():
    """Simple health check for load balancers"""
    return {"status": "ok", "service": "rag-service"}