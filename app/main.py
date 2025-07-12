from fastapi import FastAPI
import logging

from app.routes import health, query
from app.startup import lifespan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    version="1.0.0",
    description="Retrieval-Augmented Generation service with PostgreSQL vector embeddings and AWS Bedrock",
    lifespan=lifespan
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)