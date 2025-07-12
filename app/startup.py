import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import json
import time

from app.embeddings import get_embeddings
from app.bedrock import get_bedrock_client
from app.config import BEDROCK_MODEL_ID, MAX_TOKENS, TEMPERATURE

# Configure logging
logger = logging.getLogger(__name__)

# Global state to track warmup completion
_warmup_completed = False
_embeddings_loaded = False
_bedrock_initialized = False

def should_perform_warmup() -> bool:
    """Determine if warmup should be performed based on environment"""
    # Check for Fargate/ECS environment indicators
    fargate_indicators = [
        os.getenv('AWS_EXECUTION_ENV'),  # Set by ECS/Fargate
        os.getenv('ECS_CONTAINER_METADATA_URI'),  # ECS metadata
        os.getenv('ECS_CONTAINER_METADATA_URI_V4'),  # ECS metadata v4
        os.getenv('AWS_BATCH_JOB_ID'),  # AWS Batch
    ]
    
    # If any Fargate/ECS indicator is present, perform warmup
    if any(indicator for indicator in fargate_indicators):
        logger.info("🔍 Detected Fargate/ECS environment - warmup will be performed")
        return True
    
    # Check for explicit warmup override
    warmup_override = os.getenv('ENABLE_MODEL_WARMUP', '').lower()
    if warmup_override in ['true', '1', 'yes', 'on']:
        logger.info("🔍 Model warmup explicitly enabled via ENABLE_MODEL_WARMUP")
        return True
    elif warmup_override in ['false', '0', 'no', 'off']:
        logger.info("🔍 Model warmup explicitly disabled via ENABLE_MODEL_WARMUP")
        return False
    
    # Default: no warmup for Docker builds, ECR tests, local development
    logger.info("🔍 Non-production environment detected - skipping model warmup")
    return False

def is_warmup_required() -> bool:
    """Check if warmup is required based on environment"""
    return should_perform_warmup()

def is_warmup_completed() -> bool:
    """Check if warmup has been completed"""
    global _warmup_completed
    return _warmup_completed

def are_models_ready() -> bool:
    """Check if models are ready (either warmup completed or warmup not required)"""
    if not should_perform_warmup():
        # If warmup not required, models are considered ready when basic components are loaded
        return _embeddings_loaded and _bedrock_initialized
    else:
        # If warmup is required, models are ready only when warmup is completed
        return _warmup_completed

async def warmup_bedrock_model():
    """Warmup the Bedrock model to avoid ModelNotReadyException"""
    global _warmup_completed
    try:
        logger.info("🔥 Warming up Bedrock model...")
        bedrock_client = get_bedrock_client()
        
        # Send a simple test query to warm up the model
        warmup_payload = {
            "prompt": json.dumps({"messages": [{"role": "user", "content": "Hello, this is a warmup request."}]}),
            "max_tokens": 50,
            "temperature": 0.5
        }
        
        max_warmup_attempts = 3
        for attempt in range(max_warmup_attempts):
            try:
                response = bedrock_client.invoke_model(
                    modelId=BEDROCK_MODEL_ID,
                    body=json.dumps(warmup_payload),
                    contentType="application/json"
                )
                logger.info(f"✅ Bedrock model warmed up successfully on attempt {attempt + 1}")
                _warmup_completed = True
                return True
                
            except Exception as e:
                if "ModelNotReadyException" in str(e):
                    if attempt < max_warmup_attempts - 1:
                        wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                        logger.warning(f"Model not ready (attempt {attempt + 1}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning("⚠️ Model still not ready after warmup attempts")
                        _warmup_completed = False
                        return False
                else:
                    logger.warning(f"⚠️ Warmup failed with different error: {e}")
                    _warmup_completed = False
                    return False
        
    except Exception as e:
        logger.warning(f"⚠️ Bedrock warmup failed: {e}")
        _warmup_completed = False
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - conditionally preload models on startup"""
    global _embeddings_loaded, _bedrock_initialized
    
    # Startup - Preload models
    logger.info("🚀 Starting RAG Service...")
    
    # Check if we should perform warmup
    perform_warmup = should_perform_warmup()
    
    try:
        # Always preload embeddings model (lightweight)
        logger.info("📥 Loading embeddings model...")
        embeddings = get_embeddings()
        if embeddings:
            logger.info("✅ Embeddings model loaded successfully")
            _embeddings_loaded = True
        else:
            logger.warning("⚠️ Embeddings model failed to load")
            _embeddings_loaded = False
        
        # Initialize Bedrock client (lightweight)
        logger.info("🔗 Initializing Bedrock client...")
        try:
            bedrock_client = get_bedrock_client()
            logger.info("✅ Bedrock client initialized successfully")
            _bedrock_initialized = True
            
            # Conditionally warmup the model
            if perform_warmup:
                warmup_success = await warmup_bedrock_model()
                if warmup_success:
                    logger.info("🔥 Bedrock model is ready for queries")
                else:
                    logger.warning("⚠️ Bedrock model warmup incomplete - first query may be slower")
            else:
                logger.info("⏭️ Skipping Bedrock model warmup (not in production environment)")
                
        except Exception as e:
            logger.warning(f"⚠️ Bedrock client initialization failed: {e}")
            _bedrock_initialized = False
        
        if perform_warmup:
            logger.info("🎉 RAG Service startup complete - models preloaded and warmed up!")
        else:
            logger.info("🎉 RAG Service startup complete - models preloaded (warmup skipped)!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        # Continue startup even if model loading fails
    
    yield
    
    # Shutdown
    logger.info("🔄 RAG Service shutdown complete")