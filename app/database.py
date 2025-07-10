import os
import boto3
import json
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager
from botocore.exceptions import ClientError
from app.config import settings, const_fieldname_db_host, const_fieldname_db_port, const_fieldname_db_name, const_fieldname_db_user, const_fieldname_db_pass
from app.config import db_host, db_port, db_name, db_user, db_pass


async def get_db_credentials():
    """Get database credentials from AWS Secrets Manager"""
    if not settings.DB_SECRET_NAME or not settings.AWS_REGION:
        # Return None to use environment variables
        return None
    
    try:
        # Create a Secrets Manager client
        session = boto3.Session(profile_name=settings.AWS_PROFILE if settings.AWS_PROFILE else None)
        client = session.client(
            service_name='secretsmanager',
            region_name=settings.AWS_REGION
        )
        
        get_secret_value_response = client.get_secret_value(
            SecretId=settings.DB_SECRET_NAME
        )
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret
    except ClientError as e:
        print(f"Warning: Could not get credentials from AWS Secrets Manager: {e}")
        return None


async def get_database_dsn():
    """Build database DSN dynamically"""
    try:
        credentials = await get_db_credentials()
        if credentials:
            password = credentials.get('password', '')
            username = credentials.get('username', settings.DB_USER)
        else:
            # Fallback to environment variables
            password = os.getenv(const_fieldname_db_pass, db_pass)
            username = os.getenv(const_fieldname_db_user, db_user)
    except Exception as e:
        print(f"Error getting credentials: {e}")
        password = os.getenv(const_fieldname_db_pass, db_pass)
        username = os.getenv(const_fieldname_db_user, db_user)
    
    # Build DSN using the same pattern as your team's connection.py
    dsn = (
        f"host={os.getenv(const_fieldname_db_host, db_host)} "
        f"port={os.getenv(const_fieldname_db_port, db_port)} "
        f"dbname={os.getenv(const_fieldname_db_name, db_name)} "
        f"user={username} "
        f"password={password}"
    )
    
    return dsn


# Global connection pool
db_pool: AsyncConnectionPool = None


async def open_db_pool():
    """Open the database connection pool"""
    global db_pool
    
    dsn = await get_database_dsn()
    db_pool = AsyncConnectionPool(conninfo=dsn, max_size=20, min_size=5)
    
    if db_pool:
        await db_pool.open()


async def close_db_pool():
    """Close the database connection pool"""
    global db_pool
    if db_pool:
        await db_pool.close()


@asynccontextmanager
async def get_db_connection():
    """
    Acquire a connection from the async pool (matching your team's pattern)
    """
    global db_pool
    try:
        conn = await db_pool.getconn()
    except Exception as e:
        raise Exception(f"Failed to acquire DB connection: {e}")
    
    try:
        yield conn
    finally:
        await db_pool.putconn(conn)