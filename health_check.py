#!/usr/bin/env python3
"""
Health Check Script - Replicates the /health-deep endpoint functionality
Tests Database, Bedrock, and Embeddings connectivity
"""
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_bedrock_connection():
    """Test Bedrock connection - same as health_deep endpoint"""
    try:
        from app.bedrock import test_bedrock_connection
        return test_bedrock_connection()
    except Exception as e:
        print(f"❌ Bedrock test error: {e}")
        return False

def test_database_connection():
    """Test database connection - same as health_deep endpoint"""
    try:
        from app.database import test_database_connection
        return test_database_connection()
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_embeddings():
    """Test embeddings - same as health_deep endpoint"""
    try:
        from app.embeddings import is_embeddings_ready
        return is_embeddings_ready()
    except Exception as e:
        print(f"❌ Embeddings test error: {e}")
        return False

def print_header():
    """Print script header"""
    print("=" * 60)
    print("🏥 RAG SERVICE HEALTH CHECK")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Replicating: /health-deep endpoint functionality")
    print("=" * 60)

def print_test_result(component: str, status: bool, details: str = ""):
    """Print formatted test result"""
    status_icon = "✅" if status else "❌"
    status_text = "HEALTHY" if status else "UNHEALTHY"
    
    print(f"{status_icon} {component:15} | {status_text:10} | {details}")

def print_summary(overall_status: str, bedrock_status: str, database_status: str, embeddings_status: str):
    """Print final summary"""
    print("=" * 60)
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 60)
    
    overall_icon = "✅" if overall_status == "healthy" else "❌"
    print(f"{overall_icon} OVERALL STATUS: {overall_status.upper()}")
    print("-" * 60)
    print(f"   Bedrock:    {bedrock_status}")
    print(f"   Database:   {database_status}")  
    print(f"   Embeddings: {embeddings_status}")
    print("=" * 60)
    
    if overall_status == "healthy":
        print("🎉 All systems operational! RAG service is ready.")
    else:
        print("⚠️  Some components are unhealthy. Check the details above.")
        print("\n💡 Common fixes:")
        if database_status == "unhealthy":
            print("   - Database: Check connection string and credentials")
        if bedrock_status == "unhealthy":
            print("   - Bedrock: Verify AWS credentials and model permissions")
        if embeddings_status == "unhealthy":
            print("   - Embeddings: Check sentence-transformers installation")

def main():
    """Main health check function - replicates /health-deep endpoint logic"""
    
    print_header()
    
    # Test Bedrock (same logic as health_deep)
    print("\n🔍 Running health checks...")
    print("-" * 60)
    
    print("🤖 Testing Bedrock connection...")
    bedrock_healthy = test_bedrock_connection()
    bedrock_status = "healthy" if bedrock_healthy else "unhealthy"
    print_test_result("Bedrock", bedrock_healthy, "AWS Bedrock model access")
    
    # Test Database (same logic as health_deep)
    print("\n🗄️  Testing Database connection...")
    database_healthy = test_database_connection()
    database_status = "healthy" if database_healthy else "unhealthy"
    print_test_result("Database", database_healthy, "PostgreSQL + PGVector")
    
    # Test Embeddings (same logic as health_deep)
    print("\n🧠 Testing Embeddings model...")
    embeddings_healthy = test_embeddings()
    embeddings_status = "healthy" if embeddings_healthy else "unhealthy"
    print_test_result("Embeddings", embeddings_healthy, "sentence-transformers model")
    
    # Overall status (same logic as health_deep)
    overall_status = "healthy" if all([
        bedrock_status == "healthy",
        database_status == "healthy", 
        embeddings_status == "healthy"
    ]) else "unhealthy"
    
    # Print summary
    print_summary(overall_status, bedrock_status, database_status, embeddings_status)
    
    # Return appropriate exit code
    exit_code = 0 if overall_status == "healthy" else 1
    
    print(f"\n🚪 Exiting with code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚡ Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during health check: {e}")
        sys.exit(1)