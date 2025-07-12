import json
import logging
from typing import List, Tuple, Optional
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document

from app.config import BEDROCK_MODEL_ID, COLLECTION_NAME, MAX_TOKENS, TEMPERATURE, SIMILARITY_THRESHOLD
from app.bedrock import get_bedrock_client
from app.database import get_connection_string
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

class RetrievalService:
    """Service for handling document retrieval and answer generation"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.bedrock_client = None
    
    def _initialize_components(self):
        """Initialize embeddings, vectorstore, and bedrock client"""
        if not self.embeddings:
            self.embeddings = get_embeddings()
            if not self.embeddings:
                raise Exception("Embeddings model not available")
        
        if not self.vectorstore:
            connection_string = get_connection_string()
            self.vectorstore = PGVector(
                collection_name=COLLECTION_NAME,
                connection_string=connection_string,
                embedding_function=self.embeddings,
                use_jsonb=True  # Use JSONB instead of JSON for better performance
            )
        
        if not self.bedrock_client:
            self.bedrock_client = get_bedrock_client()
    
    def search_similar_documents(self, query: str, max_results: int) -> List[Tuple[Document, float]]:
        """Search for similar documents using vector similarity"""
        self._initialize_components()
        
        logger.info(f"Searching for similar documents for query: {query[:100]}...")
        
        # Search similar documents
        docs = self.vectorstore.similarity_search_with_score(query, k=max_results)
        
        # Filter documents by similarity threshold
        filtered_docs = []
        for doc, score in docs:
            # Convert distance to similarity (PGVector returns distance, lower is better)
            similarity_score = 1.0 - score if score <= 1.0 else max(0.0, 2.0 - score)
            
            if similarity_score >= SIMILARITY_THRESHOLD:
                filtered_docs.append((doc, score))
        
        logger.info(f"Found {len(filtered_docs)} documents above similarity threshold {SIMILARITY_THRESHOLD}")
        return filtered_docs
    
    def generate_answer(self, query: str, documents: List[Tuple[Document, float]]) -> str:
        """Generate answer using Bedrock model based on retrieved documents"""
        self._initialize_components()
        
        # Prepare context from documents
        context = "\n\n".join([f"Source {i+1}:\n{doc[0].page_content}" for i, doc in enumerate(documents)])
        
        # Create prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer the question based only on the context above. Be concise and cite sources."""
        
        payload = {
            "prompt": json.dumps({"messages": [{"role": "user", "content": prompt}]}),
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        logger.info("Calling Bedrock model...")
        response = self.bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json"
        )
        
        result = json.loads(response["body"].read())
        answer = result["outputs"][0]["text"].strip()
        
        logger.info("✅ Answer generated successfully")
        return answer
    
    def format_sources(self, documents: List[Tuple[Document, float]]) -> List[dict]:
        """Format document sources for response"""
        sources = []
        for i, (doc, score) in enumerate(documents):
            # Calculate similarity score for display
            similarity_score = 1.0 - score if score <= 1.0 else max(0.0, 2.0 - score)
            
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "similarity_score": round(similarity_score, 3),
                "metadata": doc.metadata,
                "source_number": i + 1
            })
        
        return sources
    
    def calculate_confidence(self, documents: List[Tuple[Document, float]]) -> float:
        """Calculate confidence score based on document similarities"""
        if not documents:
            return 0.0
        
        confidence = sum(1.0 - score for _, score in documents) / len(documents)
        return round(confidence, 3)
    
    async def process_query(self, query: str, max_results: int) -> dict:
        """Process a complete RAG query and return structured response"""
        try:
            # 1. Search similar documents
            filtered_docs = self.search_similar_documents(query, max_results)
            
            if not filtered_docs:
                logger.warning(f"No documents found above similarity threshold {SIMILARITY_THRESHOLD}")
                return {
                    "answer": "No relevant information found in the knowledge base that meets the similarity threshold.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # 2. Generate answer
            answer = self.generate_answer(query, filtered_docs)
            
            # 3. Format sources
            sources = self.format_sources(filtered_docs)
            
            # 4. Calculate confidence
            confidence = self.calculate_confidence(filtered_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

# Create a singleton instance
retrieval_service = RetrievalService()