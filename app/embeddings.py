import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Global embeddings instance
_embeddings = None

def get_embeddings():
    """Get or create embeddings instance"""
    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("✅ Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings model: {e}")
            _embeddings = None
    return _embeddings

def is_embeddings_ready():
    """Check if embeddings are ready"""
    return get_embeddings() is not None