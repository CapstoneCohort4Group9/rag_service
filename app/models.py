from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or query to search for")
    collection_name: Optional[str] = Field(None, description="Vector collection name (optional)")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score threshold")


class SourceDocument(BaseModel):
    content: str = Field(..., description="The content of the source document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    similarity_score: float = Field(..., description="Similarity score for this document")
    page_number: Optional[int] = Field(None, description="Page number if available")
    source_file: Optional[str] = Field(None, description="Source file name")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer to the query")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used")
    confidence: float = Field(..., description="Confidence score of the answer")
    query: str = Field(..., description="Original query")
    total_sources_found: int = Field(..., description="Total number of relevant sources found")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    database_status: str = Field(..., description="Database connection status")
    bedrock_status: str = Field(..., description="Bedrock model status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(default="1.0.0", description="Service version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")