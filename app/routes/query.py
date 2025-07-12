import json
import logging
from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores.pgvector import PGVector

from app.models import QueryRequest, QueryResponse
from app.config import BEDROCK_MODEL_ID, COLLECTION_NAME, MAX_TOKENS, TEMPERATURE, SIMILARITY_THRESHOLD
from app.bedrock import get_bedrock_client
from app.database import get_connection_string
from app.embeddings import get_embeddings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main RAG query endpoint"""
    try:
        embeddings = get_embeddings()
        if not embeddings:
            raise HTTPException(status_code=503, detail="Embeddings model not available")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # 1. Search similar documents
        connection_string = get_connection_string()
        vectorstore = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=connection_string,
            embedding_function=embeddings,
            use_jsonb=True  # Use JSONB instead of JSON for better performance
        )
        
        docs = vectorstore.similarity_search_with_score(request.query, k=request.max_results)
        
        # Filter documents by similarity threshold
        filtered_docs = []
        for doc, score in docs:
            # Convert distance to similarity (PGVector returns distance, lower is better)
            similarity_score = 1.0 - score if score <= 1.0 else max(0.0, 2.0 - score)
            
            if similarity_score >= SIMILARITY_THRESHOLD:
                filtered_docs.append((doc, score))
        
        if not filtered_docs:
            logger.warning(f"No documents found above similarity threshold {SIMILARITY_THRESHOLD}")
            return QueryResponse(
                answer="No relevant information found in the knowledge base that meets the similarity threshold.",
                sources=[],
                confidence=0.0
            )
        
        logger.info(f"Found {len(filtered_docs)} documents above similarity threshold {SIMILARITY_THRESHOLD}")
        
        # 2. Prepare context using filtered documents
        context = "\n\n".join([f"Source {i+1}:\n{doc[0].page_content}" for i, doc in enumerate(filtered_docs)])
        
        # 3. Generate answer with Bedrock
        bedrock_client = get_bedrock_client()
        
        prompt = f"""Context:
{context}

Question: {request.query}

Answer the question based only on the context above. Be concise and cite sources."""
        
        payload = {
            "prompt": json.dumps({"messages": [{"role": "user", "content": prompt}]}),
            "max_tokens": MAX_TOKENS,  # Use config value
            "temperature": TEMPERATURE  # Use config value
        }
        
        logger.info("Calling Bedrock model...")
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json"
        )
        
        result = json.loads(response["body"].read())
        answer = result["outputs"][0]["text"].strip()
        
        logger.info("✅ Answer generated successfully")
        
        # 4. Format sources using filtered documents
        sources = []
        for i, (doc, score) in enumerate(filtered_docs):
            # Calculate similarity score for display
            similarity_score = 1.0 - score if score <= 1.0 else max(0.0, 2.0 - score)
            
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "similarity_score": round(similarity_score, 3),
                "metadata": doc.metadata,
                "source_number": i + 1
            })
        
        # Calculate confidence based on filtered documents
        confidence = sum(1.0 - score for _, score in filtered_docs) / len(filtered_docs)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")