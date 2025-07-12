import os
import boto3
import json
import logging
from botocore.exceptions import ClientError
from langchain_community.vectorstores.pgvector import PGVector

from app.config import COLLECTION_NAME
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

def get_db_credentials():
    """Get database credentials from AWS Secrets Manager"""
    secret_name = os.getenv("DB_SECRET_NAME")
    region_name = os.getenv("AWS_REGION")
    
    if not secret_name or not region_name:
        raise ValueError("DB_SECRET_NAME and AWS_REGION must be set")
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret
    except ClientError as e:
        raise e

def get_connection_string():
    """Build database connection string dynamically"""
    # Get credentials from AWS Secrets Manager
    try:
        credentials = get_db_credentials()
        password = credentials.get('db_pass')
        username = credentials.get('db_user')
    except Exception as e:
        print(f"Warning: Could not get credentials from AWS Secrets Manager: {e}")
        print("Falling back to environment variables...")
        username = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")  # Fallback only
    
    # Get other DB config from environment
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    
    # Validate required variables
    if not all([host, database, username, password]):
        raise ValueError("Missing required database configuration")
    
    # Build connection string
    connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    return connection_string

def get_db_config():
    """Get all database configuration"""
    return {
        'host': os.getenv("DB_HOST"),
        'port': os.getenv("DB_PORT", "5432"),
        'database': os.getenv("DB_NAME"),
        'connection_string': get_connection_string()
    }

def get_database_connection_string():
    """Get database connection string (wrapper for compatibility)"""
    return get_connection_string()

def test_database_connection():
    """Test database connection"""
    try:
        connection_string = get_connection_string()
        embeddings = get_embeddings()
        
        # Simple connection test
        vectorstore = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=connection_string,
            embedding_function=embeddings if embeddings else None
        )
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False