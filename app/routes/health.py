from fastapi import APIRouter, HTTPException
import logging

from app.models import HealthResponse
from app.bedrock import test_bedrock_connection
from app.database import test_database_connection
from app.embeddings import is_embeddings_ready

# Import the startup state functions
try:
    from app.startup import are_models_ready, should_perform_warmup
except ImportError:
    # Fallback if startup module is not available
    def are_models_ready():
        return True
    def should_perform_warmup():
        return False

router = APIRouter()
logger = logging.getLogger(__name__)

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
    try:
        # Wait for models to be ready if warmup is required
        if should_perform_warmup() and not are_models_ready():
            raise HTTPException(
                status_code=503, 
                detail="Service is starting up, models not ready yet..."
            )
        
        return {"status": "ok", "service": "rag-service"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    
# Add this temporarily to monitor your fixes
@router.get("/health-debug")
async def health_debug():
    """Debug endpoint to check component status"""
    try:
        from app.startup import _embeddings_loaded, _bedrock_initialized, _warmup_completed
        
        return {
            "status": "ok",
            "service": "rag-service",
            "components": {
                "embeddings_loaded": _embeddings_loaded,
                "bedrock_initialized": _bedrock_initialized, 
                "warmup_completed": _warmup_completed,
                "warmup_required": should_perform_warmup(),
                "models_ready": are_models_ready()
            }
        }
    except Exception as e:
        return {"error": str(e)}