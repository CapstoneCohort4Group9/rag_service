import os
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")  # For local development only
ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN")  # For local development only
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