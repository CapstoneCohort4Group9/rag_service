from fastapi import FastAPI
import logging

from app.routes import health, query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="RAG Service", version="1.0.0")

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)