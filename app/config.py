import os
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
# BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
# BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")  # For local development only
# ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN")  # For local development only
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "airline_docs_pg")

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# AWS Secrets Manager (for production)
DB_SECRET_NAME = os.getenv("DB_SECRET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# RAG Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Model Configuration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "384"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

RETURN_COUNT = int(os.getenv("RETURN_COUNT", "3")) 