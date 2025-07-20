from pydantic import BaseModel
from app.config import TOP_K_RESULTS

class QueryRequest(BaseModel):
    query: str
    max_results: int = TOP_K_RESULTS  # Use config default

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

class HealthResponse(BaseModel):
    status: str
    database_status: str
    embeddings_status: str