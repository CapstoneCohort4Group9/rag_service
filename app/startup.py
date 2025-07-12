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

async def warmup_bedrock_model():
    """Warmup the Bedrock model to avoid ModelNotReadyException"""
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
                        return False
                else:
                    logger.warning(f"⚠️ Warmup failed with different error: {e}")
                    return False
        
    except Exception as e:
        logger.warning(f"⚠️ Bedrock warmup failed: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - conditionally preload models on startup"""
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
        else:
            logger.warning("⚠️ Embeddings model failed to load")
        
        # Initialize Bedrock client (lightweight)
        logger.info("🔗 Initializing Bedrock client...")
        try:
            bedrock_client = get_bedrock_client()
            logger.info("✅ Bedrock client initialized successfully")
            
            # Conditionally warmup the model
            if perform_warmup:
                logger.info("🚀 Starting Bedrock model warmup process...")
                warmup_success = await warmup_bedrock_model()
                if warmup_success:
                    logger.info("🔥 Bedrock model is ready for queries - warmup completed!")
                else:
                    logger.warning("⚠️ Bedrock model warmup incomplete - first query may be slower")
            else:
                logger.info("⏭️ Skipping Bedrock model warmup (not in production environment)")
                
        except Exception as e:
            logger.warning(f"⚠️ Bedrock client initialization failed: {e}")
        
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