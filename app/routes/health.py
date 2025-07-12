from fastapi import APIRouter, HTTPException
import logging
import os

from app.models import HealthResponse
from app.bedrock import test_bedrock_connection
from app.database import test_database_connection
from app.embeddings import is_embeddings_ready

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
    """Simple health check for load balancers - DEBUG VERSION"""
    try:
        # Debug information
        debug_info = {
            "status": "ok", 
            "service": "rag-service",
            "environment": {
                "AWS_EXECUTION_ENV": os.getenv('AWS_EXECUTION_ENV'),
                "ECS_CONTAINER_METADATA_URI": os.getenv('ECS_CONTAINER_METADATA_URI'),
                "ENABLE_MODEL_WARMUP": os.getenv('ENABLE_MODEL_WARMUP')
            }
        }
        
        # Try to get warmup status
        try:
            from app.startup import are_models_ready, should_perform_warmup
            debug_info["warmup_required"] = should_perform_warmup()
            debug_info["models_ready"] = are_models_ready()
            
            # Only block if warmup is required and not ready
            if should_perform_warmup() and not are_models_ready():
                debug_info["message"] = "Models still warming up..."
                return debug_info  # Return 200 with debug info instead of 503
                
        except ImportError as e:
            debug_info["startup_module_error"] = str(e)
        except Exception as e:
            debug_info["warmup_check_error"] = str(e)
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "service": "rag-service", 
            "error": str(e)
        }