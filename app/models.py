from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float

class HealthResponse(BaseModel):
    status: str
    database_status: str
    bedrock_status: str
    embeddings_status: str