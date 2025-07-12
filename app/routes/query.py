import json
import logging
from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores.pgvector import PGVector

from app.models import QueryRequest, QueryResponse
from app.config import BEDROCK_MODEL_ID, COLLECTION_NAME
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
        
        if not docs:
            logger.warning("No relevant documents found")
            return QueryResponse(
                answer="No relevant information found in the knowledge base.",
                sources=[],
                confidence=0.0
            )
        
        logger.info(f"Found {len(docs)} relevant documents")
        
        # 2. Prepare context
        context = "\n\n".join([f"Source {i+1}:\n{doc[0].page_content}" for i, doc in enumerate(docs)])
        
        # 3. Generate answer with Bedrock
        bedrock_client = get_bedrock_client()
        
        prompt = f"""Context:
{context}

Question: {request.query}

Answer the question based only on the context above. Be concise and cite sources."""
        
        payload = {
            "prompt": json.dumps({"messages": [{"role": "user", "content": prompt}]}),
            "max_tokens": 384,
            "temperature": 0.5
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
        
        # 4. Format sources
        sources = []
        for i, (doc, score) in enumerate(docs):
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "similarity_score": round(1.0 - score, 3),
                "metadata": doc.metadata,
                "source_number": i + 1
            })
        
        confidence = sum(1.0 - score for _, score in docs) / len(docs)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")