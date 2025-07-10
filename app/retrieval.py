import json
import time
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from app.database import get_db_connection
from app.aws_session import get_bedrock_client_with_sts
from app.models import SourceDocument, QueryResponse
from app.config import settings


class RAGRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.bedrock_client = None
        
    async def get_bedrock_client(self):
        """Lazy initialization of Bedrock client"""
        if not self.bedrock_client:
            self.bedrock_client = get_bedrock_client_with_sts()
        return self.bedrock_client
    
    async def retrieve_similar_documents(
        self, 
        query: str, 
        collection_name: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[SourceDocument]:
        """Retrieve similar documents from PostgreSQL vector store"""
        
        async with get_db_connection() as conn:
            # Build connection string for PGVector
            dsn_dict = {}
            dsn_parts = conn.info.dsn.split()
            for part in dsn_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    dsn_dict[key] = value
            
            connection_string = f"postgresql://{dsn_dict['user']}:{dsn_dict['password']}@{dsn_dict['host']}:{dsn_dict['port']}/{dsn_dict['dbname']}"
            
            # Create PGVector instance
            vectorstore = PGVector(
                collection_name=collection_name,
                connection_string=connection_string,
                embedding_function=self.embeddings,
                use_jsonb=True
            )
            
            # Perform similarity search with scores
            docs_with_scores = vectorstore.similarity_search_with_score(
                query, 
                k=top_k
            )
            
            # Filter by similarity threshold and convert to SourceDocument
            source_docs = []
            for doc, score in docs_with_scores:
                # Convert score to similarity (PGVector returns distance, lower is better)
                similarity_score = 1.0 - score if score <= 1.0 else max(0.0, 2.0 - score)
                
                if similarity_score >= similarity_threshold:
                    source_doc = SourceDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        similarity_score=round(similarity_score, 4),
                        page_number=doc.metadata.get('page', None),
                        source_file=doc.metadata.get('source', None)
                    )
                    source_docs.append(source_doc)
            
            return source_docs
    
    async def generate_answer_with_bedrock(
        self, 
        query: str, 
        context_docs: List[SourceDocument]
    ) -> Tuple[str, float]:
        """Generate answer using Bedrock model with context"""
        
        bedrock_client = await self.get_bedrock_client()
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Source {i+1} (Score: {doc.similarity_score}):\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Create system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents. 
        
Instructions:
- Answer the question using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so
- Include citations by referencing "Source X" where relevant
- Be concise but comprehensive in your response
- Maintain a professional and helpful tone"""
        
        # Create user prompt
        user_prompt = f"""Context Documents:
{context}

Question: {query}

Please provide a detailed answer based on the context above."""
        
        # Prepare messages for Bedrock (matching your team's pattern)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Construct payload (matching your team's pattern)
        bedrock_prompt_payload = {
            "messages": messages
        }
        
        payload = {
            "prompt": json.dumps(bedrock_prompt_payload),
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE
        }
        
        try:
            # Invoke Bedrock model
            response = bedrock_client.invoke_model(
                modelId=settings.BEDROCK_MODEL_ID,
                body=json.dumps(payload),
                contentType="application/json"
            )
            
            model_response = json.loads(response["body"].read())
            
            # Extract response text (matching your team's pattern)
            answer = model_response["outputs"][0]["text"].strip()
            
            # Calculate confidence based on context quality and answer length
            confidence = self._calculate_confidence(context_docs, answer)
            
            return answer, confidence
            
        except Exception as e:
            raise RuntimeError(f"Bedrock model invocation failed: {str(e)}")
    
    def _calculate_confidence(self, docs: List[SourceDocument], answer: str) -> float:
        """Calculate confidence score based on various factors"""
        if not docs or not answer:
            return 0.0
        
        # Base confidence on average similarity scores
        avg_similarity = sum(doc.similarity_score for doc in docs) / len(docs)
        
        # Adjust based on number of sources
        source_factor = min(1.0, len(docs) / 3.0)  # More sources = higher confidence
        
        # Adjust based on answer length (too short might indicate insufficient info)
        length_factor = min(1.0, len(answer.split()) / 50.0)  # Optimal around 50 words
        
        # Check if answer indicates uncertainty
        uncertainty_phrases = ["I don't know", "not enough information", "cannot determine", "unclear"]
        uncertainty_penalty = 0.3 if any(phrase in answer.lower() for phrase in uncertainty_phrases) else 0.0
        
        confidence = (avg_similarity * 0.6 + source_factor * 0.2 + length_factor * 0.2) - uncertainty_penalty
        
        return round(max(0.0, min(1.0, confidence)), 3)
    
    async def query(
        self,
        query: str,
        collection_name: str = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> QueryResponse:
        """Main query method that retrieves and generates response"""
        start_time = time.time()
        
        # Use default collection if not provided
        if not collection_name:
            collection_name = settings.COLLECTION_NAME
        
        try:
            # Step 1: Retrieve relevant documents
            source_docs = await self.retrieve_similar_documents(
                query, 
                collection_name, 
                top_k=max_results,
                similarity_threshold=similarity_threshold
            )
            
            if not source_docs:
                # No relevant documents found
                answer = "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your query or ask about topics covered in the airline regulations document."
                confidence = 0.0
            else:
                # Step 2: Generate answer using Bedrock
                answer, confidence = await self.generate_answer_with_bedrock(query, source_docs)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return QueryResponse(
                answer=answer,
                sources=source_docs,
                confidence=confidence,
                query=query,
                total_sources_found=len(source_docs),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            
            return QueryResponse(
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                query=query,
                total_sources_found=0,
                processing_time_ms=processing_time
            )


# Global retriever instance
retriever = RAGRetriever()