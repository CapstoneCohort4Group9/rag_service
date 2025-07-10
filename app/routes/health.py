from fastapi import APIRouter, HTTPException, status
from app.models import HealthResponse
from app.database import get_db_connection
from app.aws_session import get_bedrock_client_with_sts
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the RAG service and its dependencies"
)
async def health_check():
    """
    Comprehensive health check for the RAG service.
    
    Checks:
    - Database connectivity
    - Bedrock model accessibility
    - Overall service status
    """
    
    database_status = "unknown"
    bedrock_status = "unknown"
    overall_status = "unhealthy"
    
    # Check database connectivity
    try:
        async with get_db_connection() as conn:
            # Simple query to test connection
            cursor = conn.cursor()
            await cursor.execute("SELECT 1")
            result = await cursor.fetchone()
            if result and result[0] == 1:
                database_status = "healthy"
            else:
                database_status = "unhealthy"
            await cursor.close()
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        database_status = "unhealthy"
    
    # Check Bedrock connectivity
    try:
        bedrock_client = get_bedrock_client_with_sts()
        # Simple test - just check if we can create the client
        if bedrock_client:
            bedrock_status = "healthy"
        else:
            bedrock_status = "unhealthy"
    except Exception as e:
        logger.error(f"Bedrock health check failed: {str(e)}")
        bedrock_status = "unhealthy"
    
    # Determine overall status
    if database_status == "healthy" and bedrock_status == "healthy":
        overall_status = "healthy"
    elif database_status == "healthy" or bedrock_status == "healthy":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    response = HealthResponse(
        status=overall_status,
        database_status=database_status,
        bedrock_status=bedrock_status
    )
    
    # Return appropriate HTTP status code
    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.dict()
        )
    
    return response


@router.get(
    "/health/simple",
    summary="Simple health check",
    description="Simple health endpoint that returns 200 OK if service is running"
)
async def simple_health_check():
    """
    Simple health check endpoint for load balancers.
    Returns basic status without checking dependencies.
    """
    return {"status": "ok", "service": "rag-service"}