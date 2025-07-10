from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.database import open_db_pool, close_db_pool
from app.routes import query, health
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events"""
    # Startup
    await open_db_pool()
    print("🚀 RAG Service started successfully")
    print(f"📊 Database pool initialized")
    print(f"🤖 Bedrock Model: {settings.BEDROCK_MODEL_ID}")
    
    yield
    
    # Shutdown
    await close_db_pool()
    print("🔄 RAG Service shutdown complete")


app = FastAPI(
    title="RAG Service API",
    description="Retrieval-Augmented Generation service using PostgreSQL vector embeddings and AWS Bedrock",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])

@app.get("/")
async def root():
    return {
        "message": "RAG Service API",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )