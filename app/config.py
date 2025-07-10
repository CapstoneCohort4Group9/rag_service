import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "hopjetairline_db"
    DB_USER: str = "hopjetair"
    DB_PASSWORD: Optional[str] = None
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_PROFILE: Optional[str] = None
    DB_SECRET_NAME: Optional[str] = None
    
    # Bedrock Configuration
    BEDROCK_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str
    ASSUME_ROLE_ARN: str
    
    # Vector Store Configuration
    COLLECTION_NAME: str = "airline_docs_pg"
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Application Configuration
    MAX_TOKENS: int = 384
    TEMPERATURE: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Database field name constants (matching your team's pattern)
const_fieldname_db_host = "DB_HOST"
const_fieldname_db_port = "DB_PORT"
const_fieldname_db_name = "DB_NAME"
const_fieldname_db_user = "DB_USER"
const_fieldname_db_pass = "DB_PASSWORD"

# Default values (fallback)
db_host = "localhost"
db_port = 5432
db_name = "hopjetairline_db"
db_user = "hopjetair"
db_pass = ""

settings = Settings()