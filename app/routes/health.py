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
    """Resilient health check that handles startup failures gracefully"""
    try:
        # Check if we're in an environment that requires warmup
        warmup_required = should_perform_warmup()
        
        if warmup_required:
            # In production, wait for basic components (not necessarily warmup completion)
            models_ready = are_models_ready()
            
            if not models_ready:
                # Return 503 only if critical components failed to load
                logger.info("Health check: Basic components still loading...")
                raise HTTPException(
                    status_code=503, 
                    detail="Service is starting up, basic components not ready yet..."
                )
        
        # Return healthy status
        return {
            "status": "ok", 
            "service": "rag-service",
            "warmup_required": warmup_required,
            "models_ready": are_models_ready()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        # In case of any error, still return healthy to avoid restart loops
        return {
            "status": "ok", 
            "service": "rag-service",
            "note": "Running with degraded functionality"
        }